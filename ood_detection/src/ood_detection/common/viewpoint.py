"""Disagreement of the predicted mu for one world cell seen from two windows (no labels needed)."""

import numpy as np
import torch

CHUNK = 256
MIN_SHARED = 100


def rot2(yaw):
    """(..., 2, 2) rotation matrices from (...,) yaw angles."""
    c, s = torch.cos(yaw), torch.sin(yaw)
    return torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)


def to_frame(xy, pose_src, pose_dst):
    """(B, N, 2) points in pose_src's BEV frame, expressed in pose_dst's. Poses are (x, y, yaw)."""
    t = (rot2(-pose_dst[:, 2]) @ (pose_src[:, :2] - pose_dst[:, :2]).unsqueeze(-1)).squeeze(-1)
    return xy @ rot2(pose_src[:, 2] - pose_dst[:, 2]).mT + t.unsqueeze(1)


def normalized_grid(xy, grid_bounds):
    """BEV frame coordinates to grid_sample coordinates, matching _sample_path."""
    (x0, x1, _), (y0, y1, _) = grid_bounds["xbound"], grid_bounds["ybound"]
    u = 2 * (xy[..., 0] - x0) / (x1 - x0) - 1
    v = 2 * (xy[..., 1] - y0) / (y1 - y0) - 1
    return torch.stack((-v, -u), -1)


def sample_maps(maps, xy, grid_bounds):
    """(B, C, N) bilinear samples of (B, C, X, Y) maps at (B, N, 2) frame coordinates.

    Out-of-footprint points come back NaN rather than clamped to the border."""
    grid = normalized_grid(xy, grid_bounds).unsqueeze(1)
    inside = (grid.abs() <= 1).all(-1)
    out = torch.nn.functional.grid_sample(maps, grid, align_corners=True).squeeze(2)
    return out.masked_fill(~inside, torch.nan)


def partner_index(mission, k):
    """Index of the window k steps later in the same mission, else -1 (windows are in recording order)."""
    n = len(mission)
    partner = np.full(n, -1)
    ahead = np.arange(n - k) + k
    same = mission[:n - k] == mission[ahead]
    partner[:n - k] = np.where(same, ahead, -1)
    return partner


def top_frac_nanmean(d, frac, min_cells=MIN_SHARED):
    """Mean of the worst `frac` of each row's finite entries; NaN below min_cells."""
    n = torch.isfinite(d).sum(1)
    s = torch.sort(d.nan_to_num(-1.0), dim=1, descending=True).values
    k = torch.clamp((frac * n).round().long(), min=1)
    take = torch.arange(d.shape[1], device=d.device).unsqueeze(0) < k.unsqueeze(1)
    return torch.where(n >= min_cells, (s * take).sum(1) / k, torch.nan)


def disagreement(mu_maps, observed, poses, mission, centers, grid_bounds, k=10,
                 top_frac=0.1, device="cpu"):
    """(N,) mean |delta mu| over the worst `top_frac` of cells both windows observe, else NaN."""
    partner = partner_index(np.asarray(mission), k)
    poses = torch.as_tensor(np.asarray(poses), dtype=torch.float32, device=device)
    xy = torch.as_tensor(np.asarray(centers), dtype=torch.float32, device=device).reshape(-1, 2)
    out = np.full(len(mu_maps), np.nan)
    have = np.flatnonzero(partner >= 0)

    for i in range(0, len(have), CHUNK):
        a, b = have[i:i + CHUNK], partner[have[i:i + CHUNK]]
        mu_a = torch.as_tensor(mu_maps[a], dtype=torch.float32, device=device)
        seen_a = torch.as_tensor(observed[a], dtype=torch.float32, device=device)
        got = sample_maps(torch.stack([mu_a, seen_a], 1),
                          to_frame(xy.expand(len(a), -1, -1), poses[b], poses[a]), grid_bounds)
        mu_b = torch.as_tensor(mu_maps[b], dtype=torch.float32, device=device).reshape(len(b), -1)
        seen_b = torch.as_tensor(observed[b], device=device).reshape(len(b), -1)

        d = (got[:, 0] - mu_b).abs()
        out[a] = top_frac_nanmean(
            d.masked_fill(~(seen_b & (got[:, 1] > 0.99)), torch.nan), top_frac).cpu().numpy()
    return out


N_PROBES = 24
MIN_PROBES = 6


def disagreement_spread(mu_maps, observed, poses, mission, centers, grid_bounds,
                        n_probes=N_PROBES, top_frac=0.1, min_probes=MIN_PROBES,
                        device="cpu"):
    """(N, 4) spread of each cell's prediction over the next `n_probes` windows.

    Columns: (max-min, std) x (worst-decile mean, median over cells). Cells seen by fewer
    than `min_probes` viewpoints are NaN. Probes run forward only: those share the far
    cells, where the signal is."""
    mission = np.asarray(mission)
    poses = torch.as_tensor(np.asarray(poses), dtype=torch.float32, device=device)
    xy = torch.as_tensor(np.asarray(centers), dtype=torch.float32, device=device).reshape(-1, 2)
    n = len(mu_maps)
    out = np.full((n, 4), np.nan)

    for i in range(0, n, CHUNK):
        a = np.arange(i, min(i + CHUNK, n))
        seen_a = torch.as_tensor(observed[a], device=device).reshape(len(a), -1)
        base = torch.as_tensor(mu_maps[a], dtype=torch.float32, device=device).reshape(len(a), -1)
        zeroed = torch.where(seen_a, base, torch.zeros_like(base))
        total, total_sq, count = zeroed.clone(), zeroed ** 2, seen_a.float()
        hi = torch.where(seen_a, base, torch.full_like(base, -torch.inf))
        lo = torch.where(seen_a, base, torch.full_like(base, torch.inf))

        for d in range(1, n_probes + 1):
            b = a + d
            ok = (b < n) & (mission[np.clip(b, 0, n - 1)] == mission[a])
            if not ok.any():
                continue
            bi = np.where(ok, b, a)
            maps_b = torch.as_tensor(mu_maps[bi], dtype=torch.float32, device=device)
            seen_b = torch.as_tensor(observed[bi], dtype=torch.float32, device=device)
            got = sample_maps(torch.stack([maps_b, seen_b], 1),
                              to_frame(xy.expand(len(a), -1, -1), poses[a], poses[bi]),
                              grid_bounds)
            valid = (torch.as_tensor(ok, device=device).unsqueeze(1) & seen_a
                     & (got[:, 1] > 0.99) & torch.isfinite(got[:, 0]))
            v = torch.where(valid, got[:, 0], torch.zeros_like(got[:, 0]))
            total, total_sq, count = total + v, total_sq + v ** 2, count + valid.float()
            hi = torch.where(valid, torch.maximum(hi, got[:, 0]), hi)
            lo = torch.where(valid, torch.minimum(lo, got[:, 0]), lo)

        var = (total_sq - total ** 2 / count.clamp(min=1)) / (count - 1).clamp(min=1)
        enough_cell = count >= min_probes
        rng = torch.where(enough_cell, hi - lo, torch.nan)
        std = torch.where(enough_cell, var.clamp(min=0).sqrt(), torch.nan)
        for j, s in enumerate((rng, std)):
            enough = torch.isfinite(s).sum(1) >= MIN_SHARED
            out[a, 2 * j] = top_frac_nanmean(s, top_frac).cpu().numpy()
            out[a, 2 * j + 1] = torch.where(enough, torch.nanmedian(s, dim=1).values,
                                            torch.nan).cpu().numpy()
    return out
