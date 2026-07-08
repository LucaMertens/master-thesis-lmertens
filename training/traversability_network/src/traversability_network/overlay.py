"""Camera-image overlays of BEV traversability values, shared by the trainer
(wandb panels) and visualize_predictions.py (standalone CLI)."""
import cv2
import numpy as np
import torch


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


def overlay_path_on_image(rgb, states, target_trav, trav_weights, K, extrinsics, show_gated=False):
    """Ground-truth driven path, colored per-point by its labelled mu. Points with
    zero mu weight (gated) are skipped, or drawn as gray outlines with show_gated."""
    valid = torch.isfinite(states[:, 0])
    if not show_gated:
        valid &= trav_weights[:, 0] > 0
    p = states[valid].numpy()
    mu = target_trav[valid, 0].numpy()
    weighted = (trav_weights[valid, 0] > 0).numpy()
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
    uv, mu, weighted = uv[inside], mu[inside], weighted[inside]
    colors = colorize_mu(mu)
    overlay = rgb.copy()
    for (u, v), c, wgt in zip(uv, colors, weighted):
        if wgt:
            cv2.circle(overlay, (int(round(u)), int(round(v))), 5, tuple(int(x) for x in c), -1, cv2.LINE_AA)
        else:
            cv2.circle(overlay, (int(round(u)), int(round(v))), 5, (128, 128, 128), 1, cv2.LINE_AA)
    return overlay
