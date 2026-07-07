"""Qualitative check of a checkpoint: predicted mu/nu projected onto the anchor
camera image, next to the same image with the ground-truth (RHE) path colored by
its labelled mu. Samples both train and valid missions, saves PNGs locally.
"""
import argparse
import os

import cv2
import numpy as np
import torch

from .config import get_cfg
from .data.grandtour_dataset import GrandTourDataset
from .network.trainer import TrainingModule


def colorize_mu(mu):
    """mu in [0,1] -> BGR uint8, red (slippery, low mu) to green (grippy, high mu)."""
    mu = np.clip(mu, 0.0, 1.0)
    r = np.where(mu < 0.5, 255, np.round(255 * (1 - 2 * (mu - 0.5)))).astype(np.uint8)
    g = np.where(mu < 0.5, np.round(255 * 2 * mu), 255).astype(np.uint8)
    b = np.zeros_like(r)
    return np.stack([b, g, r], axis=-1)


def grid_world_centers(grid_bounds, Hd, Wd):
    """Base-frame (x, y) at z=0 for each cell of a (Hd, Wd) trav_map, matching the
    grid_sample convention in TrainingModule._sample_path."""
    xb0, xb1 = grid_bounds["xbound"][0], grid_bounds["xbound"][1]
    yb0, yb1 = grid_bounds["ybound"][0], grid_bounds["ybound"][1]
    x = xb1 - np.arange(Hd) / (Hd - 1) * (xb1 - xb0)
    y = yb1 - np.arange(Wd) / (Wd - 1) * (yb1 - yb0)
    return np.meshgrid(x, y, indexing="ij")  # each (Hd, Wd)


def project_to_cam(X, Y, K, extrinsics):
    """World (x, y, z=0) -> pixel (u, v) and in-front-of-camera mask. extrinsics is
    cam2base (4, 4), matching TrainingModule._project_path_to_cam."""
    pts = np.stack([X, Y, np.zeros_like(X)], axis=-1).reshape(-1, 3)
    R, t = extrinsics[:3, :3], extrinsics[:3, 3]
    cam = (pts - t) @ R
    pix = cam @ K.T
    uv = pix[:, :2] / np.clip(pix[:, 2:3], 1e-6, None)
    return uv.reshape(*X.shape, 2), (cam[:, 2].reshape(X.shape) > 0.3)


def overlay_bev_on_image(rgb, values, K, extrinsics, grid_bounds, stride=2, alpha=0.55):
    """values: (Hd, Wd) array in [0,1] (mu or nu). Splats colored dots at each
    grid cell's camera projection, sized by inverse depth for pseudo-perspective."""
    Hd, Wd = values.shape
    X, Y = grid_world_centers(grid_bounds, Hd, Wd)
    uv, front = project_to_cam(X[::stride, ::stride], Y[::stride, ::stride], K, extrinsics)
    vals = values[::stride, ::stride]
    h, w = rgb.shape[:2]
    inside = front & (uv[..., 0] >= 0) & (uv[..., 0] < w) & (uv[..., 1] >= 0) & (uv[..., 1] < h)
    uv, vals = uv[inside], vals[inside]
    if len(uv) == 0:
        return rgb.copy()
    # depth for radius: recompute z from the same projection (cheap, small arrays)
    pts = np.stack([X[::stride, ::stride], Y[::stride, ::stride], np.zeros_like(X[::stride, ::stride])], -1)
    z = ((pts.reshape(-1, 3) - extrinsics[:3, 3]) @ extrinsics[:3, :3])[:, 2].reshape(X[::stride, ::stride].shape)[inside]
    colors = colorize_mu(vals)
    overlay = rgb.copy()
    for (u, v), c, zc in zip(uv, colors, z):
        radius = int(np.clip(35.0 / max(zc, 0.5), 2, 14))
        cv2.circle(overlay, (int(round(u)), int(round(v))), radius, tuple(int(x) for x in c), -1, cv2.LINE_AA)
    return cv2.addWeighted(overlay, alpha, rgb, 1 - alpha, 0)


def overlay_path_on_image(rgb, states, target_trav, trav_weights, K, extrinsics):
    """Ground-truth driven path, colored per-point by its labelled mu; only points
    with nonzero mu weight (observable) are drawn."""
    valid = torch.isfinite(states[:, 0]) & (trav_weights[:, 0] > 0)
    p = states[valid].numpy()
    mu = target_trav[valid, 0].numpy()
    if len(p) == 0:
        return rgb.copy()
    pts = np.stack([p[:, 0], p[:, 1], np.zeros_like(p[:, 0])], -1)
    R, t = extrinsics[:3, :3], extrinsics[:3, 3]
    cam = (pts - t) @ R
    front = cam[:, 2] > 0.3
    pix = cam @ K.T
    uv = pix[:, :2] / np.clip(pix[:, 2:3], 1e-6, None)
    h, w = rgb.shape[:2]
    inside = front & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    uv, mu = uv[inside], mu[inside]
    colors = colorize_mu(mu)
    overlay = rgb.copy()
    for (u, v), c in zip(uv, colors):
        cv2.circle(overlay, (int(round(u)), int(round(v))), 5, tuple(int(x) for x in c), -1, cv2.LINE_AA)
    return overlay


def tensor_to_bgr(color_img):
    """(3, H, W) float [0,1] RGB -> (H, W, 3) uint8 BGR, for cv2 drawing/imwrite."""
    rgb = (color_img.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def label_panel(img, text):
    img = img.copy()
    cv2.putText(img, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def load_model(cfg_file, checkpoint, device):
    configs = get_cfg(cfg_file)
    model = TrainingModule(configs)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    if missing or unexpected:
        print(f"non-strict load: missing={missing} unexpected={unexpected}")
    model.eval().to(device)
    return configs, model


@torch.no_grad()
def render_sample(model, batch, device, grid_bounds):
    color_img, depth_voxel, intrinsics_full, intrinsics, extrinsics, states, target_trav, trav_weights, depth_target, depth_mask = batch
    b = tuple(x.unsqueeze(0).to(device) for x in (color_img, depth_voxel, intrinsics_full, intrinsics, extrinsics, depth_target))
    trav_map, _, _, _ = model.model(*b)
    trav_map = trav_map[0].cpu().numpy()  # (2, Hd, Wd)

    K, ext = intrinsics[-1].numpy(), extrinsics[-1].numpy()
    rgb = tensor_to_bgr(color_img[-1])

    mu_panel = label_panel(overlay_bev_on_image(rgb, trav_map[0], K, ext, grid_bounds), "predicted mu")
    nu_panel = label_panel(overlay_bev_on_image(rgb, trav_map[1], K, ext, grid_bounds), "predicted nu")
    gt_panel = label_panel(overlay_path_on_image(rgb, states, target_trav, trav_weights, K, ext), "labelled mu (path)")
    return np.concatenate([mu_panel, nu_panel, gt_panel], axis=1)


def main():
    parser = argparse.ArgumentParser(description="Overlay predicted/labelled traversability on the camera image.")
    parser.add_argument("--cfg_file", required=True, help="config the checkpoint was trained with")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out_dir", default="vis_predictions")
    parser.add_argument("--n_samples", type=int, default=8, help="images saved per split")
    parser.add_argument("--n_lowmu", type=int, default=4, help="of n_samples, reserved for windows with a labelled low-mu path point")
    parser.add_argument("--lowmu_thresh", type=float, default=0.5)
    parser.add_argument("--max_scan", type=int, default=200, help="cap on windows inspected per split")
    parser.add_argument("--splits", nargs="+", default=["train", "valid"], choices=["train", "valid"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    configs, model = load_model(args.cfg_file, args.checkpoint, args.device)
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    for split in args.splits:
        missions = configs.DATASET.TRAIN_MISSIONS if split == "train" else configs.DATASET.VALID_MISSIONS
        dataset = GrandTourDataset(configs, missions, train=False)
        order = rng.permutation(len(dataset))[: args.max_scan]

        n_saved, n_lowmu_saved = 0, 0
        for idx in order:
            if n_saved >= args.n_samples:
                break
            batch = dataset[int(idx)]
            target_trav, trav_weights = batch[6], batch[7]
            has_lowmu = bool(((target_trav[:, 0] < args.lowmu_thresh) & (trav_weights[:, 0] > 0)).any())
            n_other_slots = args.n_samples - args.n_lowmu
            if has_lowmu and n_lowmu_saved >= args.n_lowmu:
                continue
            if not has_lowmu and (n_saved - n_lowmu_saved) >= n_other_slots:
                continue

            panel = render_sample(model, batch, args.device, configs.MODEL.GRID_BOUNDS)
            tag = "lowmu" if has_lowmu else "random"
            fname = os.path.join(args.out_dir, f"{split}_{n_saved:03d}_{tag}_idx{idx}.png")
            cv2.imwrite(fname, panel)

            n_saved += 1
            n_lowmu_saved += int(has_lowmu)

        print(f"{split}: saved {n_saved} images ({n_lowmu_saved} with a labelled low-mu path point) -> {args.out_dir}")


if __name__ == "__main__":
    main()
