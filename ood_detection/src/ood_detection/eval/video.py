"""One video per tier: BEV Mahalanobis++ cell distance on the ground, detector scores beside it.

The model reruns for per-cell features; the printed scores come from the cache, matched
by nearest timestamp. Colours use the ID cell-distance range, numbers show ID percentiles.

    uv run python -m ood_detection.eval.video \
        --cfg training/traversability_network/configs/gt_human_mt_ineq075_f2.yaml \
        --checkpoint /datahome/lmertens/checkpoints/owbc12tm/best_epoch=11-step=11385.ckpt \
        --features-dir /datalocal/lmertens/ood_cache/owbc12tm_v2 --out-dir <dir>
"""

import argparse
import os

import cv2
import imageio.v2 as iio
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from traversability_network.adaptations import mission_ids
from traversability_network.adaptations.grandtour_dataset import GrandTourDataset
from traversability_network.adaptations.overlay import overlay_bev_on_image
from traversability_network.adaptations.visualize_predictions import tensor_to_bgr

from ood_detection.common.bev_cells import cell_centers, ground_height, visible_cells
from ood_detection.common.checkpoint import BEVFeatureHook, load_checkpoint
from ood_detection.common.splits import HELDOUT_TIER
from ood_detection.common.windows import anchor_timestamp
from ood_detection.eval.run import cell_rows, load_tier
from ood_detection.eval.scores import id_scale
from ood_detection.eval.window_scores import cached
from ood_detection.mahalanobis.detector import MahalanobisPP
from ood_detection.report.make_retiered_report import DET_LABEL, TIER_LABEL, TIERS

FPS = 5.0
CRF = 30
ALPHA = 0.45
# the detector painted on the map; the rest are numbers only
MAP_DETECTOR = "bev_mahalanobis"
PANEL = [MAP_DETECTOR, "image_mahalanobis", "dino_mahalanobis", "depth_entropy",
         "viewpoint_range_med", "modality_disagreement"]
PANEL_W, HEADER_H, LEGEND_H = 320, 56, 62
# max timestamp gap (s) to count as the same window
TOL = 0.5
FONT = cv2.FONT_HERSHEY_SIMPLEX
GREY, DIM, WHITE = (185, 185, 185), (150, 150, 150), (255, 255, 255)
CALM, ALARM = (170, 130, 70), (60, 60, 235)  # BGR: bar below and above the ID 95th


def colorize_ood(x):
    """0 (in-distribution) to 1 (far), BGR like colorize_mu."""
    return cv2.applyColorMap((np.clip(x, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)


def put(img, text, xy, scale=0.38, color=GREY):
    cv2.putText(img, text, xy, FONT, scale, color, 1, cv2.LINE_AA)


def percentile(ref, v):
    """Percentile of a score in its detector's held-out ID distribution."""
    return np.nan if not np.isfinite(v) else np.searchsorted(ref, v) / len(ref)


def reference(scores):
    """detector -> its sorted held-out ID window scores."""
    held = scores["tier"] == HELDOUT_TIER
    out = {}
    for det in PANEL:
        v = scores[f"score_{det}"][held]
        out[det] = np.sort(v[np.isfinite(v)])
    return out


def score_panel(h, tier, mission, t_rel, values, ref, scale, w=PANEL_W):
    """Side column with the window's location and every detector's score."""
    p = np.full((h, w, 3), 26, np.uint8)
    put(p, TIER_LABEL.get(tier, tier), (12, 24), 0.5, WHITE)
    put(p, f"{mission}   t = {t_rel:.0f} s", (12, 43), 0.36, DIM)
    row = (h - HEADER_H - LEGEND_H) // len(PANEL)
    bar_x, bar_w = 12, w - 128
    for i, det in enumerate(PANEL):
        y, v = HEADER_H + i * row, values[det]
        q = percentile(ref[det], v)
        on_map = det == MAP_DETECTOR
        put(p, DET_LABEL[det] + ("   [map]" if on_map else ""), (12, y + 12), 0.36,
            WHITE if on_map else GREY)
        cv2.rectangle(p, (bar_x, y + 20), (bar_x + bar_w, y + 29), (60, 60, 60), -1)
        if np.isfinite(q):
            colour = ALARM if q >= 0.95 else CALM
            cv2.rectangle(p, (bar_x, y + 20), (bar_x + int(q * bar_w), y + 29), colour, -1)
        tick = bar_x + int(0.95 * bar_w)  # the ID 95th percentile, where FPR95 is read
        cv2.line(p, (tick, y + 18), (tick, y + 31), (120, 120, 120), 1)
        text = "--" if not np.isfinite(v) else f"{v:.3g}  p{100 * q:.0f}"
        put(p, text, (bar_x + bar_w + 8, y + 29), 0.36, WHITE if on_map else GREY)

    y0 = h - LEGEND_H + 14
    put(p, "cell distance on the ground", (12, y0), 0.33, DIM)
    ramp = colorize_ood(np.linspace(0, 1, w - 24)[None])
    p[y0 + 8:y0 + 20, 12:w - 12] = np.repeat(ramp, 12, axis=0)
    put(p, f"{scale[0]:.0f}", (12, y0 + 34), 0.33, DIM)
    put(p, "ID p1 - p99", (w // 2 - 32, y0 + 34), 0.33, DIM)
    put(p, f"{scale[1]:.0f}", (w - 34, y0 + 34), 0.33, DIM)
    return p


def match(ts, want, tol=TOL):
    """(dataset index, wanted index) of the nearest dataset window within tol of each cached one."""
    order = np.argsort(ts)
    t = ts[order]
    j = np.clip(np.searchsorted(t, want), 1, len(t) - 1)
    j = np.where(np.abs(t[j - 1] - want) < np.abs(t[j] - want), j - 1, j)
    ok = np.abs(t[j] - want) <= tol
    return order[j[ok]], np.flatnonzero(ok)


@torch.no_grad()
def mission_frames(configs, model, detector, mission, rows, scores, ref, scale, device,
                   batch_size, workers):
    """Yields one annotated BGR frame per matched cached window."""
    ds = GrandTourDataset(configs, [mission], train=False, time_ranges={})
    if len(ds) == 0:
        return
    ts = np.array([anchor_timestamp(ds, i) for i in range(len(ds))])
    idx, hit = match(ts, scores["timestamp"][rows])
    if not len(idx):
        print(f"{mission}: no dataset window within {TOL}s of a cached one")
        return
    rows, t0 = rows[hit], ts.min()
    loader = DataLoader(Subset(ds, idx.tolist()), batch_size=batch_size, shuffle=False,
                        num_workers=workers)
    centers, lo, hi = cell_centers(model), *scale

    b0 = 0
    with BEVFeatureHook(model) as hook:
        for batch in loader:
            color, depth_voxel, K_full, K, ext, states = (x.to(device) for x in batch[:6])
            depth_target = batch[8].to(device)
            extra = [x.to(device) for x in batch[10:]]
            feats, _, _ = mission_ids.split(extra, *mission_ids.flags(configs))
            model.model(color, depth_voxel, K_full, K, ext, depth_target, image_feats=feats)
            observed = visible_cells(
                centers, ground_height(states, centers, configs.MODEL.GRID_BOUNDS), K, ext,
                configs.MODEL.GRID_BOUNDS, configs.MODEL.INPUT_SIZE).cpu().numpy()
            bev = hook.features.permute(0, 2, 3, 1).float().cpu().numpy()

            for b in range(color.shape[0]):
                d = detector.cell_scores(bev[b].reshape(-1, bev.shape[-1]))
                d = np.where(observed[b], d.reshape(observed.shape[1:]), np.nan)
                row = rows[b0 + b]
                frame = overlay_bev_on_image(
                    tensor_to_bgr(color[b, -1].cpu()), (d - lo) / (hi - lo),
                    K[b, -1].cpu().numpy(), ext[b, -1].cpu().numpy(), configs.MODEL.GRID_BOUNDS,
                    ground_z=model._ground_plane(states[b].cpu()),
                    depth=depth_voxel[b, -1].float().cpu().numpy() * 1e-3,
                    depth_K=K_full[b, -1].cpu().numpy(), alpha=ALPHA, colorize=colorize_ood)
                panel = score_panel(
                    frame.shape[0], str(scores["tier"][row]), mission,
                    scores["timestamp"][row] - t0,
                    {det: scores[f"score_{det}"][row] for det in PANEL}, ref, scale)
                yield cv2.hconcat([frame, panel])
            b0 += color.shape[0]


def write_video(path, frames, fps):
    writer, n = None, 0
    for frame in frames:
        if writer is None:
            writer = iio.get_writer(path, fps=fps, codec="libx264", macro_block_size=2,
                                    output_params=["-crf", str(CRF), "-preset", "veryfast",
                                                   "-pix_fmt", "yuv420p"])
        writer.append_data(cv2.cvtColor(np.ascontiguousarray(frame), cv2.COLOR_BGR2RGB))
        n += 1
    if writer is not None:
        writer.close()
    return n


def tier_rows(scores, tier, stride, max_frames):
    """Cached windows of one tier, thinned to at most max_frames."""
    rows = np.flatnonzero(scores["tier"] == tier)
    return rows[::max(stride, -(-len(rows) // max_frames))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--features-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--scores", default=None, help="window-score npz; default: in --features-dir")
    ap.add_argument("--tiers", nargs="+", default=None)
    ap.add_argument("--stride", type=int, default=4, help="cached windows between frames")
    ap.add_argument("--max-frames", type=int, default=500, help="cap per tier")
    ap.add_argument("--fps", type=float, default=FPS)
    ap.add_argument("--modes", type=int, default=3)
    ap.add_argument("--rescore", action="store_true", help="redo the cached window scores")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device = torch.device(args.device)
    configs, model = load_checkpoint(args.cfg, args.checkpoint, device)
    scores = cached(args.features_dir, args.scores, args.modes, args.rescore)
    ref = reference(scores)

    id_train = load_tier(args.features_dir, "id_train")
    detector = MahalanobisPP(args.modes, metric="mahalanobis").fit(cell_rows(id_train))
    held = load_tier(args.features_dir, HELDOUT_TIER)["cells"][::10].astype(np.float32)
    scale = id_scale(detector.cell_scores(held.reshape(-1, held.shape[-1])))
    print(f"ID cell-distance colour scale {scale[0]:.2f} to {scale[1]:.2f}")

    os.makedirs(args.out_dir, exist_ok=True)
    present = set(scores["tier"])
    for tier in (args.tiers or [t for t in [HELDOUT_TIER] + TIERS if t in present]):
        rows = tier_rows(scores, tier, args.stride, args.max_frames)
        missions = dict.fromkeys(scores["mission"][rows])
        frames = (f for m in tqdm(missions, desc=tier)
                  for f in mission_frames(configs, model, detector, m,
                                          rows[scores["mission"][rows] == m], scores, ref,
                                          scale, device, args.batch_size, args.workers))
        path = os.path.join(args.out_dir, f"{tier}.mp4")
        print(f"{tier}: {write_video(path, frames, args.fps)} of {len(rows)} frames -> {path}")


if __name__ == "__main__":
    main()
