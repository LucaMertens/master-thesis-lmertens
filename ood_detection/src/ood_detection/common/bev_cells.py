"""BEV cell geometry: terrain height, observability mask and per-window cell subsampling."""

import torch

from traversability_network.adaptations.propagate import OCCL_ABS, OCCL_FRAC

MIN_FIT_STATES = 10
RIDGE = 1e-6


def cell_centers(model):
    """(X, Y, 2) cell centres in the anchor base frame, read off the model's own voxel grid."""
    return model.model.voxels[0, :, :, :2]


def ground_height(states, centers, grid_bounds):
    """(B, X, Y) terrain height: least-squares plane through the path's ground z, clamped to its range."""
    x, y, z = states[..., 0], states[..., 1], states[..., 5]
    (x0, x1, _), (y0, y1, _) = grid_bounds["xbound"], grid_bounds["ybound"]
    ok = torch.isfinite(z) & (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)

    A = torch.stack([torch.ones_like(x), x, y], -1) * ok[..., None]
    A = torch.nan_to_num(A)
    zw = torch.nan_to_num(z) * ok
    ridge = RIDGE * torch.eye(3, device=A.device, dtype=A.dtype)
    coef = torch.linalg.solve(A.mT @ A + ridge, (A.mT @ zw.unsqueeze(-1)))

    B = states.shape[0]
    plane = (coef.view(B, 1, 1, 3) * torch.cat(
        [torch.ones_like(centers[..., :1]), centers], -1)).sum(-1)
    zf = torch.where(ok, z, torch.nan)
    lo = torch.nan_to_num(zf.nan_to_num(float("inf")).min(1).values).view(B, 1, 1)
    hi = torch.nan_to_num(zf.nan_to_num(float("-inf")).max(1).values).view(B, 1, 1)
    plane = plane.clamp(min=lo, max=hi)
    flat = torch.nan_to_num(torch.nanmedian(zf, dim=1).values).view(B, 1, 1)
    return torch.where((ok.sum(1) >= MIN_FIT_STATES).view(B, 1, 1), plane, flat)


def visible_cells(centers, ground_z, intrinsics, extrinsics, grid_bounds, image_size,
                  depth=None):
    """(B, X, Y) bool: cells inside any input frame's frustum; optional depth (mm) drops occluded cells."""
    d0, d1 = grid_bounds["dbound"][0], grid_bounds["dbound"][1]
    B, T = extrinsics.shape[:2]
    X, Y = centers.shape[:2]
    xy = centers.reshape(1, 1, X * Y, 2).expand(B, T, -1, -1)
    z_col = ground_z.reshape(B, 1, X * Y, 1).expand(B, T, -1, -1)
    pts = torch.cat([xy, z_col], dim=-1)

    cam = (pts - extrinsics[..., :3, 3].unsqueeze(2)) @ extrinsics[..., :3, :3]
    z = cam[..., 2]
    u = intrinsics[..., 0, 0:1] * cam[..., 0] / z + intrinsics[..., 0, 2:3]
    v = intrinsics[..., 1, 1:2] * cam[..., 1] / z + intrinsics[..., 1, 2:3]
    seen = ((z > d0) & (z < d1) & (u >= 0) & (u < image_size[0])
            & (v >= 0) & (v < image_size[1]))

    if depth is not None:
        H, W = depth.shape[-2:]
        un = (u * W / image_size[0]).long().clamp(0, W - 1)
        vn = (v * H / image_size[1]).long().clamp(0, H - 1)
        d = depth.reshape(B, T, -1).gather(2, vn * W + un).float() * 1e-3
        margin = torch.clamp(OCCL_FRAC * z, min=OCCL_ABS)
        seen &= ~((d > 0) & (d < z - margin))

    return seen.any(1).reshape(B, X, Y)


def sample_cells(features, mask, n_keep, generator=None):
    """(B, n_keep, C) observed cells drawn uniformly per window, plus (B,) observed counts.

    Windows with fewer observed cells are padded with unobserved ones; the count marks them."""
    B, C = features.shape[0], features.shape[1]
    flat = features.reshape(B, C, -1).transpose(1, 2)
    rank = torch.rand(flat.shape[:2], device=flat.device, generator=generator)
    rank = rank + 2.0 * ~mask.reshape(B, -1)
    idx = rank.topk(n_keep, dim=1, largest=False).indices
    return flat.gather(1, idx.unsqueeze(-1).expand(-1, -1, C)), mask.reshape(B, -1).sum(1)
