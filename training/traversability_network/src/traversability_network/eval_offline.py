"""Score finished checkpoints offline: mu confusion counts per population, no training.

    uv run python -m traversability_network.eval_offline <config.yaml> <ckpt> <out.npz>
"""
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from .adaptations import keep_recording_indices, subsample_indices, weighting
from .adaptations.grandtour_dataset import GrandTourDataset
from .config import get_cfg
from .network.dataloader import Dataset as WayfasterDataset
from .network.trainer import TrainingModule

N_BINS = 10
BIN_W = 1.0 / N_BINS
MAX_WINDOWS = 15000
WORKERS = 16
RANGE_BINS = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 20)]
WEIGHT_CACHE = "/datalocal/lmertens/{}_train_weights.npy"
# Top-down accumulator in robot frame (x forward, y left), for where blind cells sit.
MAP_X, MAP_Y, MAP_RES = (-4.0, 20.0), (-10.0, 10.0), 0.25


def frustum_mask(states, intrinsics, extrinsics, grid_bounds, image_size):
    """(B, H, T) bool: is each horizon state's ground point inside frame t's frustum.

    Column 5 is the ground z (about -0.55 m on GrandTour); without it almost nothing is visible.
    Padded (inf) states count as not visible.
    """
    b_n, h_n = states.shape[:2]
    t_n = intrinsics.shape[1]
    out = np.zeros((b_n, h_n, t_n), dtype=bool)
    for b in range(b_n):
        s = states[b].astype(np.float64)
        ok = np.isfinite(s).all(1)
        if not ok.any():
            continue
        pts = np.zeros((int(ok.sum()), 6))
        pts[:, :2] = s[ok, :2]
        pts[:, 5] = s[ok, 5]
        for t in range(t_n):
            out[b, ok, t] = weighting.visible_states(
                pts, [intrinsics[b, t]], [extrinsics[b, t]], grid_bounds, image_size,
            ).astype(bool)
    return out


class Bins:
    """Confusion counts and summed predictions per true-mu bin for one population."""

    def __init__(self):
        self.counts = np.zeros((N_BINS, N_BINS))
        self.pred_sum = np.zeros(N_BINS)
        # exact sums, the binned matrix would be off by up to a bin
        self.err = np.zeros(3)          # [n, sum |e|, sum e^2]

    def add(self, pred, true):
        if not len(true):
            return
        tb = np.clip((true / BIN_W).astype(int), 0, N_BINS - 1)
        pb = np.clip((pred / BIN_W).astype(int), 0, N_BINS - 1)
        np.add.at(self.counts, (tb, pb), 1)
        np.add.at(self.pred_sum, tb, pred)
        e = pred - true
        self.err += [len(e), np.abs(e).sum(), (e ** 2).sum()]

    def means(self):
        row = self.counts.sum(1)
        return np.where(row > 0, self.pred_sum / np.maximum(row, 1), np.nan)


class Map:
    """Per-xy-cell sums in the robot frame."""

    NX = int((MAP_X[1] - MAP_X[0]) / MAP_RES)
    NY = int((MAP_Y[1] - MAP_Y[0]) / MAP_RES)
    FIELDS = ("n", "nb", "mu", "mub", "low", "lowb", "pred", "predb")

    def __init__(self):
        self.a = {f: np.zeros(self.NX * self.NY) for f in self.FIELDS}

    def add(self, path, keep, seen, true, pred):
        xy = path[..., :2]
        ix = np.where(np.isfinite(xy).all(-1), (xy[..., 0] - MAP_X[0]) / MAP_RES, -1)
        iy = np.where(np.isfinite(xy).all(-1), (xy[..., 1] - MAP_Y[0]) / MAP_RES, -1)
        ok = keep & (ix >= 0) & (ix < self.NX) & (iy >= 0) & (iy < self.NY)
        flat = (ix.astype(int) * self.NY + iy.astype(int))[ok]
        blind = ~seen[ok]
        low = (true[ok] < 0.3).astype(float)
        for f, v in (("n", 1.0), ("mu", true[ok]), ("low", low), ("pred", pred[ok])):
            np.add.at(self.a[f], flat, v)
            np.add.at(self.a[f + "b"], flat, np.where(blind, v, 0.0))

    def out(self, tag):
        return {f"{tag}_map_{f}": v.reshape(self.NX, self.NY) for f, v in self.a.items()}


def dataset_spec(cfg):
    """(class, train split, valid split) for whichever dataset the config points at."""
    if getattr(cfg.DATASET, "TRAIN_MISSIONS", None):
        return GrandTourDataset, cfg.DATASET.TRAIN_MISSIONS, cfg.DATASET.VALID_MISSIONS
    return WayfasterDataset, cfg.DATASET.TRAIN_DIRS, cfg.DATASET.VALID_DIRS


def trained_indices(ds, cfg):
    """Train windows this run saw, or None if all. Splits train into seen vs held-out."""
    rec_frac = getattr(cfg.DATASET, "TRAIN_RECORDING_FRACTION", 1.0)
    frac = getattr(cfg.DATASET, "TRAIN_FRACTION", 1.0)
    seed = getattr(cfg.DATASET, "TRAIN_SUBSET_SEED", 0)
    if rec_frac < 1.0:
        return keep_recording_indices(ds.bag_sizes, rec_frac, seed)
    if frac < 1.0:
        return subsample_indices(list(range(ds.bag_sizes[-1])), frac, seed)
    return None


@torch.no_grad()
def score(module, cfg, dirs, weights, train, max_windows=MAX_WINDOWS, indices=None, tag=None):
    cls = dataset_spec(cfg)[0]
    ds = cls(cfg, dirs, weights=weights, train=False)
    if indices is not None:
        ds.indices = indices
    # Deterministic stride to cap scoring time
    if len(ds.indices) > max_windows:
        step = len(ds.indices) / max_windows
        ds.indices = [ds.indices[int(i * step)] for i in range(max_windows)]
    print(f"  scoring {len(ds.indices)} windows", flush=True)
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=WORKERS,
                        multiprocessing_context="forkserver" if WORKERS else None)
    pops, xymap = {}, Map()
    seen_frames = np.zeros(3)             # [kept cells, seen by last frame, seen by any]
    gb, isz = module.grid_bounds, cfg.MODEL.INPUT_SIZE
    for batch in loader:
        # K_nat is native resolution, K is INPUT_SIZE-scaled (the frustum test needs K)
        color, voxel, K_nat, K, ext, path, target, w, dtarget, dmask, *extra = batch
        trav_map = module.model(color.cuda(), voxel.cuda(), K_nat.cuda(), K.cuda(),
                               ext.cuda(), dtarget.cuda())[0]
        trav, mask = module._sample_path(path.cuda(), trav_map)
        keep = mask.bool().cpu().numpy() & (w[..., 0].numpy() > 0)
        per_frame = frustum_mask(path.numpy(), K.numpy(), ext.numpy(), gb, isz)
        seen = per_frame.any(2)
        pred = trav[..., 0].float().cpu().numpy()
        true = target[..., 0].numpy()
        p = path.numpy()
        rng = np.sqrt(p[..., 0] ** 2 + p[..., 1] ** 2)
        seen_frames += [keep.sum(), (keep & per_frame[..., -1]).sum(), (keep & seen).sum()]

        def acc(name, m):
            pops.setdefault(name, Bins()).add(pred[m], true[m])

        xymap.add(p, keep, seen, true, pred)
        acc("all", keep)
        acc("vis", keep & seen)
        acc("blind", keep & ~seen)
        # Range-stratified, to compare blind and visible cells at fixed range
        for lo, hi in RANGE_BINS:
            band = keep & (rng >= lo) & (rng < hi)
            acc(f"r{lo:g}_{hi:g}_vis", band & seen)
            acc(f"r{lo:g}_{hi:g}_blind", band & ~seen)
    tag = tag or ("train" if train else "valid")
    kept, last, any_ = seen_frames
    print(f"  {tag}: {100 * last / kept:.1f}% of scored cells visible in the current "
          f"frame, {100 * any_ / kept:.1f}% in any of {cfg.MODEL.TIME_LENGTH} frames",
          flush=True)
    return {f"{tag}_{k}_counts": v.counts for k, v in pops.items()} | \
           {f"{tag}_{k}_means": v.means() for k, v in pops.items()} | \
           {f"{tag}_{k}_err": v.err for k, v in pops.items()} | \
           {f"{tag}_seen_frames": seen_frames} | xymap.out(tag)


def slope(counts, means):
    keep = (counts.sum(1) > 0) & ~np.isnan(means)
    if keep.sum() < 2:
        return float("nan")
    x = (np.arange(N_BINS)[keep] + 0.5) * BIN_W
    y = means[keep]
    return float(((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum())


def main(cfg_path, ckpt_path, out):
    cfg = get_cfg(cfg_path)
    module = TrainingModule(cfg)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    module.load_state_dict(ck["state_dict"], strict=False)
    module = module.cuda().eval()
    print(f"loaded {ckpt_path} epoch={ck.get('epoch')} step={ck.get('global_step')}", flush=True)

    # Weights from the full train set (cached, building takes minutes), same for every checkpoint
    cls, train_split, valid_split = dataset_spec(cfg)
    cache = WEIGHT_CACHE.format(cls.__name__)
    if os.path.exists(cache):
        weights = np.load(cache)
    else:
        full = get_cfg(cfg_path)
        full.DATASET.TRAIN_FRACTION = 1.0
        weights = cls(full, dataset_spec(full)[1], train=True).weights
        np.save(cache, weights)

    probe = cls(cfg, train_split, weights=weights, train=False)
    seen = trained_indices(probe, cfg)
    if seen is None:
        splits = [(False, valid_split, None, "valid"),
                  (True, train_split, None, "train")]
    else:
        unseen = sorted(set(probe.indices) - set(seen))
        splits = [(False, valid_split, None, "valid"),
                  (True, train_split, seen, "train"),
                  (True, train_split, unseen, "heldout")]
    del probe

    res = {"map_extent": np.array([*MAP_X, *MAP_Y, MAP_RES])}
    for train, dirs, idx, tag in splits:
        res |= score(module, cfg, dirs, weights, train, indices=idx, tag=tag)
        keys = ["all", "vis", "blind"] + [f"r{lo:g}_{hi:g}_{v}"
                                          for lo, hi in RANGE_BINS
                                          for v in ("vis", "blind")]
        for k in keys:
            c, m = res[f"{tag}_{k}_counts"], res[f"{tag}_{k}_means"]
            row = c.sum(1)
            n, ae, se = res[f"{tag}_{k}_err"]
            i, j = np.indices(c.shape)
            print(f"  {tag:5s} {k:14s} cells={int(row.sum()):8d} "
                  f"low%={100 * row[:3].sum() / max(row.sum(), 1):5.1f} "
                  f"slope={slope(c, m):6.3f} "
                  f"untrav_recall={c[:3, :3].sum() / max(row[:3].sum(), 1):.3f} "
                  f"mae={ae / max(n, 1):.3f} mse={se / max(n, 1):.3f} "
                  f"bin1={c[np.abs(i - j) <= 1].sum() / max(row.sum(), 1):.3f}", flush=True)
    np.savez(out, **res)
    print("wrote", out)


if __name__ == "__main__":
    main(*sys.argv[1:4])
