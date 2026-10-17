"""Render predicted-mu BEV overlaid on the camera feed as one mp4 over all missions."""

import argparse
import glob
import os
import re

import cv2
import imageio.v2 as iio
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from shared_utils.paths import OUTPUT_ROOT

from . import mission_ids
from .extract_mu_predictions import load_from_checkpoint
from .grandtour_dataset import GrandTourDataset
from .overlay import colorize_mu, overlay_bev_on_image
from .visualize_predictions import tensor_to_bgr

CKPT_ROOT = "/datahome/lmertens/checkpoints"


def resolve_checkpoint(run_id, which="best"):
    """Path of the run's best checkpoint, or the archived one at a given step."""
    pattern = "best_*.ckpt" if which == "best" else f"val_*step={which}.ckpt"
    matches = sorted(glob.glob(os.path.join(CKPT_ROOT, run_id, pattern)))
    if not matches:
        raise SystemExit(f"no checkpoint matching {pattern} under {run_id}")
    return matches[-1]


def legend_panel(h, w=160):
    """Vertical mu color-scale legend."""
    panel = np.full((h, w, 3), 30, np.uint8)
    bar_w, bar_h = 24, int(h * 0.6)
    x0, y0 = 18, (h - bar_h) // 2
    ramp = np.linspace(1.0, 0.0, bar_h, dtype=np.float32)[:, None]
    panel[y0 : y0 + bar_h, x0 : x0 + bar_w] = np.repeat(colorize_mu(ramp), bar_w, axis=1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(panel, "mu", (x0, y0 - 12), font, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(panel, "1.0", (x0 + bar_w + 6, y0 + 10), font, 0.4, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(panel, "traversable", (x0 + bar_w + 6, y0 + 26), font, 0.38, (170, 255, 170), 1, cv2.LINE_AA)
    cv2.putText(panel, "0.0", (x0 + bar_w + 6, y0 + bar_h - 20), font, 0.4, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(panel, "untraversable", (x0 + bar_w + 6, y0 + bar_h - 4), font, 0.38, (170, 170, 255), 1, cv2.LINE_AA)
    return panel


def _caption(img, text, font_scale=0.42):
    """Subtitle-style label in the top-left corner."""
    img = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, 1)
    pad = 6
    x1, y1 = tw + 2 * pad, th + baseline + 2 * pad
    roi = img[:y1, :x1]
    img[:y1, :x1] = cv2.addWeighted(roi, 0.45, np.zeros_like(roi), 0.55, 0)
    cv2.putText(img, text, (pad, pad + th), font, font_scale, (230, 230, 230), 1, cv2.LINE_AA)
    return img


@torch.no_grad()
def mission_frames(configs, model, mission, device, stride, limit, batch_size, workers, caption, alpha):
    """BGR frames (camera + BEV overlay + caption + legend) for one mission, in temporal order."""
    ds = GrandTourDataset(configs, [mission], train=False)
    if len(ds) == 0:
        print(f"{mission}: no samples, skipping")
        return
    indices = list(range(0, len(ds), stride))
    if limit:
        indices = indices[:limit]
    loader = DataLoader(Subset(ds, indices), batch_size=batch_size, shuffle=False, num_workers=workers)

    n = 0
    for batch in loader:
        color, depth_voxel, K_full, K, ext, states, target_trav, trav_weights, depth_target, _, *extra = (
            x.to(device) for x in batch
        )
        image_feats, _, _ = mission_ids.split(extra, *model.extra_flags)
        trav_map, _, _, _ = model.model(color, depth_voxel, K_full, K, ext, depth_target, image_feats=image_feats)
        mu = trav_map[:, 0].cpu().numpy()

        for b in range(color.shape[0]):
            rgb = tensor_to_bgr(color[b, -1].cpu())
            K_b, ext_b = K[b, -1].cpu().numpy(), ext[b, -1].cpu().numpy()
            depth_m = depth_voxel[b, -1].float().cpu().numpy() * 1e-3
            K_full_b = K_full[b, -1].cpu().numpy()
            ground_z = model._ground_plane(states[b].cpu())
            frame = overlay_bev_on_image(
                rgb, mu[b], K_b, ext_b, model.grid_bounds, ground_z=ground_z, depth=depth_m, depth_K=K_full_b,
                alpha=alpha,
            )
            frame = _caption(frame, f"{mission}  -  {caption}")
            frame = cv2.hconcat([frame, legend_panel(frame.shape[0])])
            yield frame[: frame.shape[0] // 2 * 2, : frame.shape[1] // 2 * 2]  # even dims for the encoder
            n += 1
    print(f"{mission}: {n} frames")


def render_video(out_path, fps, frames):
    writer = None
    n = 0
    for frame in frames:
        if writer is None:
            writer = iio.get_writer(
                out_path, fps=fps, codec="libx264", macro_block_size=2,
                output_params=["-crf", "20", "-preset", "medium", "-pix_fmt", "yuv420p"],
            )
        writer.append_data(cv2.cvtColor(np.ascontiguousarray(frame), cv2.COLOR_BGR2RGB))
        n += 1
    if writer is None:
        print("no frames, nothing written")
        return
    writer.close()
    print(f"wrote {n} frames -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", default="owbc12tm", help="wandb run id; default is the ineq0.75 run (gt_human_mt_ineq075_f2)")
    ap.add_argument("--checkpoint", default=None, help="explicit checkpoint path, overrides --run-id/--which")
    ap.add_argument("--which", default="best", help="'best' or an archived step number")
    ap.add_argument("--missions", nargs="+", default=None, help="default: the checkpoint's own VALID_MISSIONS")
    ap.add_argument("--out", default=None, help=f"default: {OUTPUT_ROOT}/videos/<run_id>/valid_missions.mp4")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--alpha", type=float, default=0.4, help="BEV overlay opacity")
    ap.add_argument("--stride", type=int, default=1, help="subsample dataset indices")
    ap.add_argument("--limit", type=int, default=None, help="cap frames per mission, for a short test render")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    checkpoint = args.checkpoint or resolve_checkpoint(args.run_id, args.which)
    device = torch.device(args.device)
    configs, model = load_from_checkpoint(checkpoint, device)

    missions = args.missions or list(configs.DATASET.VALID_MISSIONS)
    run_tag = args.run_id or configs.TAG
    out_path = args.out or os.path.join(OUTPUT_ROOT, "videos", run_tag, "valid_missions.mp4")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    epoch = re.search(r"epoch=(\d+)", os.path.basename(checkpoint))
    caption = f"{configs.TAG}  epoch {epoch.group(1)}" if epoch else configs.TAG

    def frames():
        for mission in missions:
            yield from mission_frames(
                configs, model, mission, device,
                args.stride, args.limit, args.batch_size, args.workers, caption, args.alpha,
            )

    render_video(out_path, args.fps, frames())


if __name__ == "__main__":
    main()
