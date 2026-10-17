"""Loss-weight adaptations: LDS-alpha tempering, observability gate, visibility/occlusion gate."""
import numpy as np
import torch

from .propagate import OCCL_ABS, OCCL_FRAC


def mu_match_mask(target_mu, keep_prob):
    """Per-cell Bernoulli keep to match a target mu histogram (keep_prob per 0.1 bin). Training only."""
    bins = torch.clamp((target_mu * 10).long(), 0, 9)
    return torch.rand_like(target_mu) < keep_prob[bins]


def lds_weights(values, alpha):
    """Inverse-density LDS weights tempered by alpha (1.0 = original 1/density)."""
    return (1.0 / values) ** alpha


def observability_gate(trav_weights, mu_observability, nu_observability):
    """Multiply mu/nu weight columns by their RHE observability (0 at standstill)."""
    trav_weights[:, 0] *= mu_observability
    trav_weights[:, 1] *= nu_observability
    return trav_weights


def visible_states(states, intrinsics_list, extrinsics_list, grid_bounds, image_size,
                   depth_img_list=None):
    """1.0 per state whose ground point is in at least one input frustum (within dbound), else 0.0.

    Intrinsics must be scaled to the input size. With depth_img_list (native-res mm),
    occluded points do not count."""
    d0, d1 = grid_bounds["dbound"][0], grid_bounds["dbound"][1]
    pts = np.column_stack((states[:, :2], states[:, 5]))
    visible = np.zeros(len(states), dtype=bool)
    depth_img_list = depth_img_list or [None] * len(intrinsics_list)
    for K, ext, depth_img in zip(intrinsics_list, extrinsics_list, depth_img_list):
        cam = (pts - ext[:3, 3]) @ ext[:3, :3]
        z = cam[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = K[0, 0] * cam[:, 0] / z + K[0, 2]
            v = K[1, 1] * cam[:, 1] / z + K[1, 2]
        in_frustum = (
            (z > d0) & (z < d1)
            & (u >= 0) & (u < image_size[0])
            & (v >= 0) & (v < image_size[1])
        )
        if depth_img is not None and in_frustum.any():
            us, vs = np.where(in_frustum, u, 0), np.where(in_frustum, v, 0)
            un = np.clip((us * depth_img.shape[1] / image_size[0]).astype(int),
                         0, depth_img.shape[1] - 1)
            vn = np.clip((vs * depth_img.shape[0] / image_size[1]).astype(int),
                         0, depth_img.shape[0] - 1)
            d = depth_img[vn, un].astype(np.float64) * 1e-3
            margin = np.maximum(OCCL_ABS, OCCL_FRAC * z)
            occluded = np.isfinite(d) & (d > 0) & (d < z - margin)
            in_frustum &= ~occluded
        visible |= in_frustum
    return visible.astype(np.float64)


def apply_visibility_gate(trav_weights, states, intrinsics_list, extrinsics_list,
                          grid_bounds, image_size, occlusion_gate, depth_img_list):
    """Zero the weight of horizon cells no input frame can see."""
    seen = visible_states(
        states, intrinsics_list, extrinsics_list, grid_bounds, image_size,
        depth_img_list if occlusion_gate else None,
    )
    trav_weights *= seen[:, None]
    return trav_weights
