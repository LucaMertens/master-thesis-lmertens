"""Sanity render of auto-generated W-RIZZ pairs on their anchor frames, plus pair statistics."""
import argparse
import os

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from ..config import get_cfg
from . import rizz
from .grandtour_dataset import GrandTourDataset

# BGR as in W-RIZZ Fig. 2: blue = more traversable, red = less, yellow = equal
MORE, LESS, EQUAL = (255, 90, 0), (40, 40, 255), (0, 220, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def project(states, K, ext, image_size):
    """Base-frame states (N, >=6) -> anchor-image pixels and in-view mask (column 5 is ground height)."""
    pts = np.stack([states[:, 0], states[:, 1], states[:, 5]], -1)
    cam = (pts - ext[:3, 3]) @ ext[:3, :3]
    pix = cam @ K.T
    uv = pix[:, :2] / np.clip(pix[:, 2:3], 1e-6, None)
    w, h = image_size
    ok = (cam[:, 2] > 0.3) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    return uv, ok


def frame_bgr(color_img, b):
    """Anchor (last) frame of window b as a BGR uint8 image."""
    img = color_img[b, -1].float().clamp(0, 1).numpy().transpose(1, 2, 0)
    return np.ascontiguousarray((img[..., ::-1] * 255).astype(np.uint8))


def crosshair(img, uv, color, label):
    u, v = int(round(uv[0])), int(round(uv[1]))
    cv2.drawMarker(img, (u, v), color, cv2.MARKER_CROSS, 24, 2, cv2.LINE_AA)
    cv2.circle(img, (u, v), 10, color, 2, cv2.LINE_AA)
    for c, t in ((( 0, 0, 0), 3), (color, 1)):
        cv2.putText(img, label, (u + 13, v - 12), FONT, 0.5, c, t, cv2.LINE_AA)


def caption(width, text, height=30):
    strip = np.full((height, width, 3), 245, np.uint8)
    cv2.putText(strip, text, (8, 20), FONT, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
    return strip


def pair_panel(color_img, uv, ok, mu, ba, ha, bb, hb, t, image_size):
    """One frame for an intra-window pair, two side by side for a cross-window one; None if out of view."""
    if not (ok[ba, ha] and ok[bb, hb]):
        return None
    ca = EQUAL if t == 0 else (MORE if t < 0 else LESS)
    cb = EQUAL if t == 0 else (LESS if t < 0 else MORE)
    la, lb = f"{mu[ba, ha]:.2f}", f"{mu[bb, hb]:.2f}"
    if ba == bb:
        img = frame_bgr(color_img, ba)
        pa, pb = uv[ba, ha], uv[bb, hb]
        cv2.line(img, tuple(np.round(pa).astype(int)), tuple(np.round(pb).astype(int)),
                 (255, 255, 255), 1, cv2.LINE_AA)
        crosshair(img, pa, ca, la)
        crosshair(img, pb, cb, lb)
        kind = "intra"
    else:
        left, right = frame_bgr(color_img, ba), frame_bgr(color_img, bb)
        crosshair(left, uv[ba, ha], ca, la)
        crosshair(right, uv[bb, hb], cb, lb)
        gap = np.full((left.shape[0], 6, 3), 245, np.uint8)
        img = np.hstack([left, gap, right])
        kind = "cross"
    rel = {-1: "a > b", 0: "a = b", 1: "a < b"}[int(t)]
    text = f"{kind}  |  mu a {la}  b {lb}  |  t = {int(t):+d}  ({rel})"
    return np.vstack([img, caption(img.shape[1], text)])


def pick_panels(pairs, n_intra, n_cross):
    """Indices of pairs to render: inequality first, then a tie, per pair kind."""
    (ba, ha), (bb, hb), t = pairs
    out = []
    for cross, want in ((False, n_intra), (True, n_cross)):
        sel = ((ba != bb) == cross).nonzero().squeeze(1)
        ineq = [i for i in sel.tolist() if t[i] != 0]
        eq = [i for i in sel.tolist() if t[i] == 0]
        out += ineq[: want - 1] + eq[:1]
    return out


def render_pairs(batch, cfg, out_path, n_intra=4, n_cross=4, seed=0):
    color_img, _, _, intrinsics, extrinsics, states, target, weights = batch[:8]
    gb = cfg.MODEL.GRID_BOUNDS
    x, y = states[..., 0], states[..., 1]
    valid = ((x >= gb["xbound"][0]) & (x <= gb["xbound"][1])
             & (y >= gb["ybound"][0]) & (y <= gb["ybound"][1]) & (weights[..., 0] > 0))

    gen = torch.Generator().manual_seed(seed)
    kw = dict(deadband=cfg.TRAINING.RIZZ_DEADBAND, tie_band=cfg.TRAINING.RIZZ_TIE_BAND,
              cross_frac=cfg.TRAINING.RIZZ_CROSS_FRACTION,
              min_sep=cfg.TRAINING.RIZZ_MIN_SEP, ineq_frac=cfg.TRAINING.RIZZ_INEQ_FRACTION)
    pairs = rizz.sample_pairs(target[..., 0], valid, states[..., :2],
                              n_pairs=cfg.TRAINING.RIZZ_PAIRS, generator=gen, **kw)

    B = states.shape[0]
    size = cfg.MODEL.INPUT_SIZE
    uv, ok = zip(*[project(states[b].numpy(), intrinsics[b, -1].numpy(),
                           extrinsics[b, -1].numpy(), size) for b in range(B)])
    uv, ok = np.stack(uv), np.stack(ok)
    mu = target[..., 0].numpy()

    (ba, ha), (bb, hb), t = pairs
    panels = []
    for i in pick_panels(pairs, n_intra + 3, n_cross + 3):
        panel = pair_panel(color_img, uv, ok, mu, int(ba[i]), int(ha[i]),
                           int(bb[i]), int(hb[i]), int(t[i]), size)
        if panel is not None:
            panels.append(panel)
    panels = panels[: n_intra + n_cross]
    if not panels:
        raise SystemExit("no sampled pair had both points inside the anchor frame")

    width = max(p.shape[1] for p in panels)
    rows = [np.pad(p, ((0, 0), (0, width - p.shape[1]), (0, 0)), constant_values=245)
            for p in panels]
    sep = np.full((8, width, 3), 245, np.uint8)
    grid = np.vstack([r for p in rows for r in (p, sep)])
    cv2.imwrite(out_path, grid)
    print(f"wrote {out_path}  ({len(panels)} pairs)")
    return dict(valid=valid, uv=uv, ok=ok, mu=mu, kw=kw, states=states, target=target)


def render_stats(ctx, cfg, out_path, n_pairs=40000, seed=1):
    """Population view of the pair draw: label histogram, pair balance, camera visibility."""
    import matplotlib.pyplot as plt

    valid, uv, ok, mu, kw, states, target = (
        ctx[k] for k in ("valid", "uv", "ok", "mu", "kw", "states", "target"))
    cell_mu = mu[valid.numpy()]

    gen = torch.Generator().manual_seed(seed)
    (ba, ha), (bb, hb), t = rizz.sample_pairs(
        target[..., 0], valid, states[..., :2], n_pairs=n_pairs, generator=gen, **kw)
    t = t.numpy()
    cross = (ba != bb).numpy()
    dmu = (mu[bb, hb] - mu[ba, ha])
    seen = np.concatenate([ok[ba, ha], ok[bb, hb]])

    fig, axes = plt.subplots(1, 4, figsize=(15.5, 3.4))

    axes[0].hist(cell_mu, bins=np.linspace(0, 1, 41), color="0.35")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Label mu at supervised cells")
    axes[0].set_ylabel("Cells")
    axes[0].set_title(f"n = {len(cell_mu)}, share below 0.3: {(cell_mu < 0.3).mean():.2%}")

    labels = ["a > b", "tie", "a < b"]
    intra = [int(((t == v) & ~cross).sum()) for v in (-1, 0, 1)]
    cross_n = [int(((t == v) & cross).sum()) for v in (-1, 0, 1)]
    pos = np.arange(3)
    axes[1].bar(pos, intra, 0.6, label="intra", color="0.35")
    axes[1].bar(pos, cross_n, 0.6, bottom=intra, label="cross", color="0.7")
    axes[1].set_xticks(pos, labels)
    axes[1].set_xlabel("Ordinal label")
    axes[1].set_ylabel("Pairs")
    axes[1].legend(frameon=False)
    axes[1].set_title(f"inequality {np.mean(t != 0):.0%}, cross {cross.mean():.0%}")

    axes[2].hist(np.abs(dmu), bins=np.linspace(0, 1, 41), color="0.35")
    axes[2].axvline(kw["deadband"], color="crimson", lw=1)
    axes[2].set_yscale("log")
    axes[2].set_xlabel("|mu difference| within a pair")
    axes[2].set_ylabel("Pairs")
    axes[2].set_title(f"deadband {kw['deadband']}")

    axes[3].bar([0, 1], [seen.mean(), 1 - seen.mean()], 0.6, color=["0.35", "0.7"])
    axes[3].set_xticks([0, 1], ["in view", "out of view"])
    axes[3].set_ylabel("Share of pair endpoints")
    axes[3].set_title("Anchor-frame visibility")

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cfg_file", required=True)
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--missions", nargs="*", default=None,
                   help="override TRAIN_MISSIONS, e.g. to load fewer bags")
    p.add_argument("--batch", type=int, default=0, help="which batch of the loader")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cfg = get_cfg(args.cfg_file)
    missions = args.missions or cfg.DATASET.TRAIN_MISSIONS
    ds = GrandTourDataset(cfg, missions, train=True)
    torch.manual_seed(args.seed)
    loader = DataLoader(ds, batch_size=cfg.TRAINING.BATCHSIZE, shuffle=True,
                        num_workers=0, drop_last=True)
    for i, batch in enumerate(loader):
        if i == args.batch:
            break

    os.makedirs(args.out, exist_ok=True)
    ctx = render_pairs(batch, cfg, os.path.join(args.out, "rizz_pairs.png"), seed=args.seed)
    render_stats(ctx, cfg, os.path.join(args.out, "rizz_pair_stats.png"))


if __name__ == "__main__":
    main()
