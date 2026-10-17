"""extract_dlio output must match the ap20 total-station (ATE method from odometry_fallback.ipynb)."""

import numpy as np
import pytest
from shared_utils.paths import ROS1_ROOT
from data_labeling_pipeline.ros1_transforms import get_static_transform
from grandtour_extraction.anchor import apply_rigid_2d, nearest_idx, read_point_topic, weighted_rigid_fit_2d
from grandtour_extraction.extract_dlio import extract_dlio_odom
from rosbags.rosbag1 import Reader
from scipy.spatial.transform import Rotation as R

MISSION = ROS1_ROOT / "ETH-1"


def _ate(t_src, p_src, t_ap, p_ap):
    """3D residual: xy from the shared yaw+translation fit, z from a constant offset."""
    lo, hi = max(t_src[0], t_ap[0]), min(t_src[-1], t_ap[-1])
    m = (t_ap >= lo) & (t_ap <= hi)
    j = nearest_idx(t_ap[m], t_src)
    src, dst = p_src[j], p_ap[m]
    yaw, tx, ty = weighted_rigid_fit_2d(src[:, :2], dst[:, :2])
    pred_xy = apply_rigid_2d(src[:, :2], yaw, tx, ty)
    pred_z = src[:, 2] + (dst[:, 2].mean() - src[:, 2].mean())
    pred = np.column_stack([pred_xy, pred_z])
    return float(np.sqrt(((pred - dst) ** 2).sum(1).mean()))


def test_dlio_extraction_matches_ap20_ground_truth():
    """extract_dlio base trajectory, projected to the prism, has sub-meter ATE vs ap20."""
    if not MISSION.exists() or not list(MISSION.glob("*ap20_prism*")):
        pytest.skip("Reference dataset / ap20 GT not available.")

    odom = extract_dlio_odom(MISSION)
    t = odom[:, 0]
    p_base = odom[:, 1:4]
    r_base = R.from_quat(odom[:, 4:8])

    # base pose -> prism point
    t_bp = get_static_transform(MISSION, "base", "prism").translation
    p_prism = p_base + r_base.apply(t_bp)
    # gravity-level with the trajectory's own orientations
    up = r_base.apply([0, 0, 1.0]).mean(0)
    level = R.align_vectors([[0, 0, 1]], [up / np.linalg.norm(up)])[0]
    p_prism = level.apply(p_prism)

    ap_bag = list(MISSION.glob("*ap20_prism*"))[0]
    with Reader(ap_bag) as r:
        ap_topic = next(c.topic for c in r.connections)
    t_ap, p_ap = read_point_topic(ap_bag, ap_topic)

    ate = _ate(t, p_prism, t_ap, p_ap)
    print(f"\nETH-1 dlio extraction ATE vs ap20: {ate:.3f} m")
    assert ate < 1.0, f"dlio extraction ATE too large: {ate:.3f} m (expected ~0.4 m)"
