"""Camera-image overlays of BEV traversability values."""
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
    """Base-frame (x, y) per cell of a (Hd, Wd) trav_map (grid_sample convention)."""
    xb0, xb1 = grid_bounds["xbound"][0], grid_bounds["xbound"][1]
    yb0, yb1 = grid_bounds["ybound"][0], grid_bounds["ybound"][1]
    x = xb1 - np.arange(Hd) / (Hd - 1) * (xb1 - xb0)
    y = yb1 - np.arange(Wd) / (Wd - 1) * (yb1 - yb0)
    return np.meshgrid(x, y, indexing="ij")  # each (Hd, Wd)


def ground_plane_z(X, Y, ground_z):
    """ground_z is a constant or (a, b, c, zlo, zhi): z = a*x + b*y + c clamped to [zlo, zhi], anchor frame."""
    if isinstance(ground_z, tuple):
        a, b, c, zlo, zhi = ground_z
        return np.clip(a * X + b * Y + c, zlo, zhi)
    return np.full_like(X, ground_z)


def project_to_cam(X, Y, K, extrinsics, ground_z=0.0):
    """World (x, y, ground z) -> pixel (u, v) and in-front mask. extrinsics is cam2base (4, 4)."""
    pts = np.stack([X, Y, ground_plane_z(X, Y, ground_z)], axis=-1).reshape(-1, 3)
    R, t = extrinsics[:3, :3], extrinsics[:3, 3]
    cam = (pts - t) @ R
    pix = cam @ K.T
    uv = pix[:, :2] / np.clip(pix[:, 2:3], 1e-6, None)
    return uv.reshape(*X.shape, 2), (cam[:, 2].reshape(X.shape) > 0.3)


def cell_ground_z(depth, K, extrinsics, grid_bounds, Hd, Wd, pix_stride=2):
    """Visible-surface height per BEV cell: 25th percentile z of the unprojected depth image, NaN without returns."""
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
                         ground_z=0.0, depth=None, depth_K=None, slack=1.0,
                         colorize=colorize_mu):
    """Splat dots for (Hd, Wd) values in [0,1] (NaN skipped), sized by inverse depth.

    With a depth image, dots sit on the visible surface and cells far above ground_z are skipped."""
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
    draw = (np.isfinite(z) & np.isfinite(vals) & (z > 0.3)
            & (uv[..., 0] >= 0) & (uv[..., 0] < w) & (uv[..., 1] >= 0) & (uv[..., 1] < h))
    colors = colorize(np.nan_to_num(vals))
    overlay = rgb.copy()
    for (u, v), c, zc, ok in zip(uv.reshape(-1, 2), colors.reshape(-1, 3), z.ravel(), draw.ravel()):
        if not ok:
            continue
        radius = int(np.clip(35.0 / max(zc, 0.5), 2, 14))
        cv2.circle(overlay, (int(round(u)), int(round(v))), radius, tuple(int(x) for x in c), -1, cv2.LINE_AA)
    return cv2.addWeighted(overlay, alpha, rgb, 1 - alpha, 0)


def overlay_path_on_image(rgb, states, target_trav, trav_weights, K, extrinsics, show_gated=False):
    """Driven path colored by labelled mu; gated points are skipped or drawn gray with show_gated."""
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


def surface_heights(depth, K, extrinsics, grid_bounds, Hd, Wd, ground_z=0.0, slack=1.0):
    """Height each BEV cell is drawn at; NaN without depth or far above the ground plane."""
    X, Y = grid_world_centers(grid_bounds, Hd, Wd)
    plane = ground_plane_z(X, Y, ground_z)
    seen = cell_ground_z(depth, K, extrinsics, grid_bounds, Hd, Wd)
    return np.where(np.isfinite(seen) & (seen <= plane + slack), seen, np.nan)


def corner_heights(z):
    """(Hd+1, Wd+1) node heights, the mean of the cells meeting there (no seams between quads)."""
    pad = np.full((z.shape[0] + 2, z.shape[1] + 2), np.nan)
    pad[1:-1, 1:-1] = z
    q = np.stack([pad[:-1, :-1], pad[:-1, 1:], pad[1:, :-1], pad[1:, 1:]])
    n = np.isfinite(q).sum(0)
    return np.where(n > 0, np.where(np.isfinite(q), q, 0).sum(0) / np.maximum(n, 1), np.nan)


def overlay_bev_quads(rgb, values, cell_z, K, extrinsics, grid_bounds, alpha=0.5,
                      colorize=colorize_mu):
    """Paint each BEV cell as a filled quad on the terrain surface, far cells first."""
    Hd, Wd = values.shape
    X, Y = grid_world_centers(grid_bounds, Hd, Wd)
    rx = (grid_bounds["xbound"][1] - grid_bounds["xbound"][0]) / (Hd - 1)
    ry = (grid_bounds["ybound"][1] - grid_bounds["ybound"][0]) / (Wd - 1)
    xs = np.concatenate([[X[0, 0] + rx / 2], X[:, 0] - rx / 2])
    ys = np.concatenate([[Y[0, 0] + ry / 2], Y[0] - ry / 2])
    nodes = np.stack([*np.meshgrid(xs, ys, indexing="ij"), corner_heights(cell_z)], -1)

    cam = (nodes.reshape(-1, 3) - extrinsics[:3, 3]) @ extrinsics[:3, :3]
    z = cam[:, 2].reshape(nodes.shape[:2])
    uv = ((cam @ K.T)[:, :2] / np.clip(cam[:, 2:3], 1e-6, None)).reshape(*nodes.shape[:2], 2)
    quad = np.stack([uv[:-1, :-1], uv[:-1, 1:], uv[1:, 1:], uv[1:, :-1]], axis=2)
    quad_z = np.stack([z[:-1, :-1], z[:-1, 1:], z[1:, 1:], z[1:, :-1]], axis=2)

    h, w = rgb.shape[:2]
    draw = (np.isfinite(cell_z) & np.isfinite(values) & (quad_z > 0.3).all(-1)
            & (quad[..., 0].max(-1) >= 0) & (quad[..., 0].min(-1) < w)
            & (quad[..., 1].max(-1) >= 0) & (quad[..., 1].min(-1) < h))
    colors = colorize(np.nan_to_num(values))
    order = np.argsort(-quad_z.mean(-1)[draw])  # painter's algorithm, far to near
    painted = rgb.copy()
    for q, c in zip(quad[draw][order], colors[draw][order]):
        # 1/16 px coordinates, whole-pixel corners open seams
        cv2.fillConvexPoly(painted, np.round(q * 16).astype(np.int32),
                           tuple(int(x) for x in c), cv2.LINE_8, 4)
    return cv2.addWeighted(painted, alpha, rgb, 1 - alpha, 0)
