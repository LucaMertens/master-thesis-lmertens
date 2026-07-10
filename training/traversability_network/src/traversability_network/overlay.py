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
    """Base-frame (x, y) for each cell of a (Hd, Wd) trav_map, matching the
    grid_sample convention in TrainingModule._sample_path."""
    xb0, xb1 = grid_bounds["xbound"][0], grid_bounds["xbound"][1]
    yb0, yb1 = grid_bounds["ybound"][0], grid_bounds["ybound"][1]
    x = xb1 - np.arange(Hd) / (Hd - 1) * (xb1 - xb0)
    y = yb1 - np.arange(Wd) / (Wd - 1) * (yb1 - yb0)
    return np.meshgrid(x, y, indexing="ij")  # each (Hd, Wd)


def ground_plane_z(X, Y, ground_z):
    """ground_z is either a constant height or (a, b, c, zlo, zhi): plane
    coefficients z = a*x + b*y + c clamped to [zlo, zhi], all in the anchor frame
    (0.0 = base origin, half a meter above the actual ground on ANYmal)."""
    if isinstance(ground_z, tuple):
        a, b, c, zlo, zhi = ground_z
        return np.clip(a * X + b * Y + c, zlo, zhi)
    return np.full_like(X, ground_z)


def project_to_cam(X, Y, K, extrinsics, ground_z=0.0):
    """World (x, y, ground z) -> pixel (u, v) and in-front-of-camera mask.
    extrinsics is cam2base (4, 4), matching TrainingModule._project_path_to_cam."""
    pts = np.stack([X, Y, ground_plane_z(X, Y, ground_z)], axis=-1).reshape(-1, 3)
    R, t = extrinsics[:3, :3], extrinsics[:3, 3]
    cam = (pts - t) @ R
    pix = cam @ K.T
    uv = pix[:, :2] / np.clip(pix[:, 2:3], 1e-6, None)
    return uv.reshape(*X.shape, 2), (cam[:, 2].reshape(X.shape) > 0.3)


def cell_ground_z(depth, K, extrinsics, grid_bounds, Hd, Wd, pix_stride=2):
    """Visible-surface height per BEV cell: 25th percentile z of the metric depth
    image (H, W) unprojected into the anchor frame. NaN where a cell has no returns.
    K must match the depth resolution."""
    d = depth[::pix_stride, ::pix_stride]
    vs, us = np.nonzero(d > 0.1)
    z = d[vs, us]
    cam = np.stack([(us * pix_stride - K[0, 2]) / K[0, 0] * z,
                    (vs * pix_stride - K[1, 2]) / K[1, 1] * z, z], 1)
    p = cam @ extrinsics[:3, :3].T + extrinsics[:3, 3]
    xb, yb = grid_bounds["xbound"], grid_bounds["ybound"]
    i = np.round((xb[1] - p[:, 0]) * (Hd - 1) / (xb[1] - xb[0])).astype(int)
    j = np.round((yb[1] - p[:, 1]) * (Wd - 1) / (yb[1] - yb[0])).astype(int)
    ok = (i >= 0) & (i < Hd) & (j >= 0) & (j < Wd)
    cell, pz = i[ok] * Wd + j[ok], p[ok, 2]
    if len(pz) == 0:
        return np.full((Hd, Wd), np.nan, np.float32)
    order = np.lexsort((pz, cell))
    cell, pz = cell[order], pz[order]
    starts = np.searchsorted(cell, np.arange(Hd * Wd))
    counts = np.diff(np.append(starts, len(cell)))
    sel = np.minimum(starts + (0.25 * np.maximum(counts - 1, 0)).astype(int), len(pz) - 1)
    return np.where(counts > 0, pz[sel], np.nan).reshape(Hd, Wd)


def overlay_bev_on_image(rgb, values, K, extrinsics, grid_bounds, stride=2, alpha=0.55,
                         ground_z=0.0, depth=None, depth_K=None, slack=1.0):
    """values: (Hd, Wd) array in [0,1] (mu or nu). Splats colored dots at each grid
    cell's camera projection, sized by inverse depth for pseudo-perspective. With a
    metric depth image the dots sit on the visible terrain per cell and cells whose
    surface is far above the ground_z plane are skipped (canopy, trunks, occluded);
    without one everything is drawn on the ground_z plane."""
    Hd, Wd = values.shape
    X, Y = grid_world_centers(grid_bounds, Hd, Wd)
    Z = ground_plane_z(X, Y, ground_z)
    if depth is not None:
        Zd = cell_ground_z(depth, depth_K, extrinsics, grid_bounds, Hd, Wd)
        Z = np.where(np.isfinite(Zd) & (Zd <= Z + slack), Zd, np.nan)
    Xs, Ys, Zs = X[::stride, ::stride], Y[::stride, ::stride], Z[::stride, ::stride]
    vals = values[::stride, ::stride]
    pts = np.stack([Xs, Ys, Zs], -1).reshape(-1, 3)
    cam = (pts - extrinsics[:3, 3]) @ extrinsics[:3, :3]
    z = cam[:, 2].reshape(Xs.shape)
    pix = cam @ K.T
    uv = (pix[:, :2] / np.clip(pix[:, 2:3], 1e-6, None)).reshape(*Xs.shape, 2)
    h, w = rgb.shape[:2]
    draw = np.isfinite(z) & (z > 0.3) & (uv[..., 0] >= 0) & (uv[..., 0] < w) & (uv[..., 1] >= 0) & (uv[..., 1] < h)
    colors = colorize_mu(vals)
    overlay = rgb.copy()
    for (u, v), c, zc, ok in zip(uv.reshape(-1, 2), colors.reshape(-1, 3), z.ravel(), draw.ravel()):
        if not ok:
            continue
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
    z = p[:, 5] if p.shape[1] > 5 else np.zeros_like(p[:, 0])
    pts = np.stack([p[:, 0], p[:, 1], z], -1)
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
