"""HDR over the human W-RIZZ pairs, scored offline from a checkpoint."""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from . import mission_ids
from .grandtour_dataset import GrandTourDataset
from .human_pairs import HumanPairs, PairTable, batch_scores
from .rizz import HEADLINE_TAU, balanced_hdr, hdr_counts

# thresholds on the pair difference; 0 is scale-free
TAUS = (0.0, 0.1, HEADLINE_TAU, 0.5)
# grid for the best-case threshold
TAU_SCAN = np.round(np.arange(0.0, 0.505, 0.005), 3)


def valid_split(configs, missions=None):
    """Valid-mission dataset with one human occurrence per annotation, in its annotated window."""
    missions = list(missions or configs.DATASET.VALID_MISSIONS)
    ranges = getattr(configs.DATASET, "VALID_TIME_RANGES", None) or {}
    ds = GrandTourDataset(configs, missions, train=False, time_ranges=ranges)
    table = PairTable(configs.DATASET.HUMAN_INDEX_ROOT)
    ds.human = HumanPairs(table, ds, configs.MODEL.GRID_BOUNDS, k=1, ineq_fraction=0.0,
                          seed=configs.SEED, trainable_only=False)
    return ds, table


def _annotated_windows(ds):
    """Positions of windows carrying an annotation (skipping the rest is ~6x faster)."""
    wanted = {ds.bag_sizes[b] + w for b, w in ds.human.window_tasks}
    return [i for i, w in enumerate(ds.indices) if w in wanted]


def _to_device(batch, device):
    def move(x):
        return {k: v.to(device) for k, v in x.items()} if isinstance(x, dict) else x.to(device)
    return [move(x) for x in batch]


@torch.no_grad()
def endpoint_scores(configs, model, ds, table, device, batch_size=8, workers=8):
    """(sum, count) of the predicted score of every annotated endpoint, over the split."""
    loader = DataLoader(Subset(ds, _annotated_windows(ds)), batch_size=batch_size,
                        shuffle=False, num_workers=workers)
    flags = list(mission_ids.flags(configs))
    flags[2] = True  # the dataset emits human blocks whatever the run trained on
    acc = torch.zeros(len(table), 2, device=device)
    cnt = torch.zeros_like(acc)
    for batch in loader:
        color, voxel, k_full, k_inv, ext, _, _, _, depth_target, _, *extra = _to_device(batch, device)
        image_feats, _, human = mission_ids.split(extra, *flags)
        trav_map, _, _, _ = model.model(color, voxel, k_full, k_inv, ext, depth_target,
                                        image_feats=image_feats)
        pair, side, _, _, score, _, _ = batch_scores(trav_map, human, configs.MODEL.GRID_BOUNDS)
        flat = pair * 2 + side
        acc.view(-1).index_add_(0, flat, score.float())
        cnt.view(-1).index_add_(0, flat, torch.ones_like(score, dtype=torch.float))
    return acc.cpu().numpy(), cnt.cpu().numpy()


def scored_pairs(table, acc, cnt):
    """Per-pair endpoint scores, for the pairs both of whose endpoints were scored."""
    idx = np.flatnonzero((cnt > 0).all(-1))
    p = acc[idx] / cnt[idx]
    missions = np.array([table.missions(i) for i in idx]).reshape(-1, 2)
    return dict(pair=np.array(table.ids)[idx], p_a=p[:, 0], p_b=p[:, 1],
                t=table.t[idx].astype(float), kind=table.kind[idx],
                source=table.source[idx], eligible=table.hdr_eligible[idx],
                mission_a=missions[:, 0], mission_b=missions[:, 1])


def _counts(p_a, p_b, t, taus):
    def to_t(v):
        return torch.as_tensor(np.asarray(v, dtype=np.float32))
    return hdr_counts(to_t(p_a), to_t(p_b), to_t(t), torch.as_tensor(list(taus)))


def _rates(block):
    """(all, tie, inequality) disagreement rate from one threshold's (3, 2) block."""
    def rate(c):
        return float(c[0] / c[1]) if c[1] > 0 else float("nan")
    return [rate(block[j]) for j in range(3)] + [float(balanced_hdr(block))]


def summary(p_a, p_b, t, taus=TAUS):
    """Disagreement rates of one checkpoint on one pair population.

    Headline is the balanced rate (54% of answers are ties, so the plain rate mostly
    measures how often the model says "tie"). ``hdr_balanced_best`` uses the best tau."""
    counts = _counts(p_a, p_b, t, taus)
    out = {"n_pairs": float(len(t)), "n_ineq": float((np.asarray(t) != 0).sum()),
           "spread": float(np.std(np.concatenate([p_a, p_b])))}
    for i, tau in enumerate(taus):
        a, eq, ineq, bal = _rates(counts[i])
        out.update({f"hdr_{tau:g}": a, f"hdr_eq_{tau:g}": eq, f"hdr_ineq_{tau:g}": ineq,
                    f"hdr_balanced_{tau:g}": bal})
    scan = _counts(p_a, p_b, t, TAU_SCAN)
    curve = np.array([float(balanced_hdr(scan[i])) for i in range(len(TAU_SCAN))])
    out["hdr_balanced_best"] = float(curve.min())
    out["tau_best"] = float(TAU_SCAN[int(curve.argmin())])
    return out
