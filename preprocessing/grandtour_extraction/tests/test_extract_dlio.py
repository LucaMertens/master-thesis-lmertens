"""Ground-truth anchor: extract_dlio output must match the ap20 total-station.

Reuses the prism-projection + yaw/translation ATE method validated in
odometry_fallback.ipynb (dlio median ATE ~0.38 m vs ap20 GT).
"""

from typing import Any

import numpy as np
import pytest
from config.paths import ROS1_ROOT
from data_labeling_pipeline.ros1_transforms import get_static_transform
from grandtour_extraction.extract_dlio import extract_dlio_odom
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore
from scipy.spatial.transform import Rotation as R

MISSION = ROS1_ROOT / "ETH-1"


def _read_point(bag_path, topic):
    ts = get_typestore(Stores.ROS1_NOETIC)
    rows = []
    with Reader(bag_path) as r:
        conns = [c for c in r.connections if c.topic == topic]
        for conn, _, raw in r.messages(conns):
            m: Any = ts.deserialize_ros1(raw, conn.msgtype)
            t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
            p = m.point if hasattr(m, "point") else m.pose.position
            rows.append([t, p.x, p.y, p.z])
    a = np.asarray(rows, np.float64)
    a = a[np.argsort(a[:, 0])]
    return a[:, 0], a[:, 1:4]


def _nearest_idx(t_src, t_ref):
    j = np.clip(np.searchsorted(t_ref, t_src), 1, len(t_ref) - 1)
    return np.where(t_src - t_ref[j - 1] < t_ref[j] - t_src, j - 1, j)


def _ate(t_src, p_src, t_ap, p_ap):
    lo, hi = max(t_src[0], t_ap[0]), min(t_src[-1], t_ap[-1])
    m = (t_ap >= lo) & (t_ap <= hi)
    j = _nearest_idx(t_ap[m], t_src)
    src, dst = p_src[j], p_ap[m]
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    th = np.arctan2(
        (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).sum(),
        (a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1]).sum(),
    )
    c, s = np.cos(th), np.sin(th)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    e = np.linalg.norm((Rz @ src.T).T + (mu_d - Rz @ mu_s) - dst, axis=1)
    return np.sqrt((e**2).mean())


def test_dlio_extraction_matches_ap20_ground_truth():
    """extract_dlio base trajectory, projected to the prism, has sub-meter ATE vs ap20."""
    if not MISSION.exists() or not list(MISSION.glob("*ap20_prism*")):
        pytest.skip("Reference dataset / ap20 GT not available.")

    odom = extract_dlio_odom(MISSION)  # dlio_map->base, N x 8: t,x,y,z,qw,qx,qy,qz
    t = odom[:, 0]
    p_base = odom[:, 1:4]
    r_base = R.from_quat(odom[:, [5, 6, 7, 4]])  # wxyz -> xyzw

    # project base pose onto the physical prism point the total station observes
    t_bp = get_static_transform(MISSION, "base", "prism").translation
    p_prism = p_base + r_base.apply(t_bp)
    # gravity-level using the source's own orientations (matches the notebook method)
    up = r_base.apply([0, 0, 1.0]).mean(0)
    level = R.align_vectors([[0, 0, 1]], [up / np.linalg.norm(up)])[0]
    p_prism = level.apply(p_prism)

    ap_bag = list(MISSION.glob("*ap20_prism*"))[0]
    with Reader(ap_bag) as r:
        ap_topic = next(c.topic for c in r.connections)
    t_ap, p_ap = _read_point(ap_bag, ap_topic)

    ate = _ate(t, p_prism, t_ap, p_ap)
    print(f"\nETH-1 dlio extraction ATE vs ap20: {ate:.3f} m")
    assert ate < 1.0, f"dlio extraction ATE too large: {ate:.3f} m (expected ~0.4 m)"
