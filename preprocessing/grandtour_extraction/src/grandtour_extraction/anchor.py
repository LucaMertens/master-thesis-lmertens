"""Anchor a mission's DLIO trajectory into a shared geodetic ENU frame.

Rigid 2D fit (yaw + translation) of DLIO, projected to the CPT7 lever arm, onto low-covariance
navsatfix fixes in ENU around one shared origin. ap20 can only validate DLIO's shape, not the anchor.
"""

import pathlib
from typing import Any

import numpy as np
from data_labeling_pipeline.ros1_transforms import get_static_transform
from grandtour_extraction.extract_cpt7 import extract_cpt7_navsatfix
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore
from scipy.spatial.transform import Rotation as R

# position_covariance[0] (m^2) above this is a degraded fix (RTK float/standalone under canopy).
# Dropped instead of downweighted: degraded fixes are biased, not just noisy.
# Separates good (<0.03 typical) from degraded (>0.15) missions.
NAVSATFIX_COV_THRESHOLD = 0.05
MIN_TRUSTED_FIXES = 200

_WGS84_A = 6378137.0
_WGS84_E2 = 6.69437999014e-3


def ecef(lat, lon, alt):
    lat, lon = np.radians(lat), np.radians(lon)
    n = _WGS84_A / np.sqrt(1 - _WGS84_E2 * np.sin(lat) ** 2)
    return np.stack([
        (n + alt) * np.cos(lat) * np.cos(lon),
        (n + alt) * np.cos(lat) * np.sin(lon),
        (n * (1 - _WGS84_E2) + alt) * np.sin(lat),
    ], -1)


def enu(lat, lon, alt, origin):
    """lat/lon/alt (deg, deg, m) -> east/north/up around a shared origin."""
    lat0, lon0, alt0 = origin
    d = ecef(lat, lon, alt) - ecef(lat0, lon0, alt0)
    s0, c0 = np.sin(np.radians(lat0)), np.cos(np.radians(lat0))
    s1, c1 = np.sin(np.radians(lon0)), np.cos(np.radians(lon0))
    r = np.array([[-s1, c1, 0], [-s0 * c1, -s0 * s1, c0], [c0 * c1, c0 * s1, s0]])
    return d @ r.T


def enu_to_lonlat(east, north, origin):
    """Linearized inverse of enu() (horizontal only); error is under 1 cm over a site, enough for map tiles."""
    lat0, lon0, _ = origin
    s = np.sin(np.radians(lat0))
    m_per_deg_lat = _WGS84_A * (1 - _WGS84_E2) / (1 - _WGS84_E2 * s**2) ** 1.5 * np.pi / 180
    m_per_deg_lon = _WGS84_A / np.sqrt(1 - _WGS84_E2 * s**2) * np.cos(np.radians(lat0)) * np.pi / 180
    return lon0 + east / m_per_deg_lon, lat0 + north / m_per_deg_lat


def compute_shared_origin(ros1_dirs) -> tuple:
    """Mean lat/lon/alt over all missions' fixes: the shared ENU origin."""
    llh = []
    for ros1_dir in ros1_dirs:
        try:
            llh.append(extract_cpt7_navsatfix(ros1_dir)[:, 1:4])
        except (FileNotFoundError, ValueError):
            continue
    return tuple(np.concatenate(llh).mean(0))


def read_point_topic(bag_path, topic):
    """(t, xyz) of a Point/PoseStamped-like topic, sorted by time."""
    typestore = get_typestore(Stores.ROS1_NOETIC)
    rows = []
    with Reader(bag_path) as reader:
        conns = [c for c in reader.connections if c.topic == topic]
        for conn, _, raw in reader.messages(conns):
            m: Any = typestore.deserialize_ros1(raw, conn.msgtype)
            t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
            p = m.point if hasattr(m, "point") else m.pose.position
            rows.append([t, p.x, p.y, p.z])
    a = np.asarray(rows, np.float64)
    a = a[np.argsort(a[:, 0])]
    return a[:, 0], a[:, 1:4]


def nearest_idx(t_src, t_ref):
    """Index into t_ref nearest each t_src, t_ref assumed sorted."""
    j = np.clip(np.searchsorted(t_ref, t_src), 1, len(t_ref) - 1)
    return np.where(t_src - t_ref[j - 1] < t_ref[j] - t_src, j - 1, j)


def weighted_rigid_fit_2d(src_xy, dst_xy, weights=None):
    """Closed-form yaw+translation fit of src onto dst in xy (Kabsch if unweighted)."""
    w = np.ones(len(src_xy)) if weights is None else np.asarray(weights, np.float64)
    w = w / w.sum()
    mu_s = (w[:, None] * src_xy).sum(0)
    mu_d = (w[:, None] * dst_xy).sum(0)
    a, b = src_xy - mu_s, dst_xy - mu_d
    yaw = np.arctan2(
        (w * (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])).sum(),
        (w * (a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1])).sum(),
    )
    c, s = np.cos(yaw), np.sin(yaw)
    r2 = np.array([[c, -s], [s, c]])
    t = mu_d - r2 @ mu_s
    return float(yaw), float(t[0]), float(t[1])


def apply_rigid_2d(xy, yaw, tx, ty):
    c, s = np.cos(yaw), np.sin(yaw)
    r2 = np.array([[c, -s], [s, c]])
    return xy @ r2.T + np.array([tx, ty])


def rigid_rmse_2d(src_xy, dst_xy, yaw, tx, ty):
    pred = apply_rigid_2d(src_xy, yaw, tx, ty)
    return float(np.sqrt(((pred - dst_xy) ** 2).sum(1).mean()))


def _overlap_match(t_a, t_b):
    """Mask of t_a inside t_b's time range, and nearest indices into t_b for those."""
    lo, hi = max(t_a[0], t_b[0]), min(t_a[-1], t_b[-1])
    m = (t_a >= lo) & (t_a <= hi)
    j = nearest_idx(t_a[m], t_b)
    return m, j


def _load_dlio(extracted_dir):
    dlio = np.load(pathlib.Path(extracted_dir) / "dlio_odom.npy")
    return dlio[:, 0], dlio[:, 1:4], R.from_quat(dlio[:, 4:8])


def fit_dlio_to_ap20(extracted_dir: pathlib.Path, ros1_dir: pathlib.Path) -> dict | None:
    """Independent DLIO -> ap20 fit, gravity-leveled with DLIO's orientation. None without ap20 bag or overlap.

    ap20 has no global frame, so this validates DLIO's shape only. Two ap20-fit trajectories
    cross at short range by construction (KAEB-2/3: 1154 m apart by CPT7, cross at 8 mm).
    """
    extracted_dir, ros1_dir = pathlib.Path(extracted_dir), pathlib.Path(ros1_dir)
    ap20_bags = list(ros1_dir.glob("*ap20_prism*.bag"))
    if not ap20_bags:
        return None
    t_dlio, p_dlio, r_dlio = _load_dlio(extracted_dir)
    with Reader(ap20_bags[0]) as r:
        ap_topic = next(c.topic for c in r.connections)
    t_ap, p_ap = read_point_topic(ap20_bags[0], ap_topic)

    lever_prism = get_static_transform(ros1_dir, "base", "prism").translation
    p_prism = p_dlio + r_dlio.apply(lever_prism)
    up = r_dlio.apply([0, 0, 1.0]).mean(0)
    level = R.align_vectors([[0, 0, 1]], [up / np.linalg.norm(up)])[0]
    p_prism = level.apply(p_prism)

    ma, ja = _overlap_match(t_ap, t_dlio)
    if ma.sum() == 0:
        return None
    src_xy, dst_xy = p_prism[ja, :2], p_ap[ma, :2]
    yaw, tx, ty = weighted_rigid_fit_2d(src_xy, dst_xy)
    return dict(
        yaw=yaw, tx=tx, ty=ty, rmse=rigid_rmse_2d(src_xy, dst_xy, yaw, tx, ty),
        n=int(ma.sum()), coverage=float(ma.mean()),
    )


def anchor_mission(extracted_dir: pathlib.Path, ros1_dir: pathlib.Path, origin: tuple) -> dict:
    """DLIO -> shared-ENU anchor for one mission.

    origin must be shared across missions (compute_shared_origin). Fits CPT7 fixes below
    NAVSATFIX_COV_THRESHOLD, weighted by 1/cov. If anchor_ok is False, yaw/tx/ty must not be used.
    Also runs fit_dlio_to_ap20 as validation.
    """
    extracted_dir, ros1_dir = pathlib.Path(extracted_dir), pathlib.Path(ros1_dir)
    t_dlio, p_dlio, r_dlio = _load_dlio(extracted_dir)

    cpt7 = extract_cpt7_navsatfix(ros1_dir)
    cpt7_enu = enu(cpt7[:, 1], cpt7[:, 2], cpt7[:, 3], origin)
    lever_cpt7 = get_static_transform(ros1_dir, "base", "cpt7_imu").translation
    p_point = p_dlio + r_dlio.apply(lever_cpt7)

    m, j = _overlap_match(cpt7[:, 0], t_dlio)
    src_xy, dst_xy = p_point[j, :2], cpt7_enu[m, :2]
    cov = cpt7[m, 4]
    trusted = cov < NAVSATFIX_COV_THRESHOLD
    n_trusted = int(trusted.sum())

    out = dict(
        mission=extracted_dir.name, yaw=0.0, tx=0.0, ty=0.0,
        rmse_cpt7=np.nan, n_cpt7=n_trusted, frac_cpt7_trusted=trusted.mean(),
        anchor_ok=n_trusted >= MIN_TRUSTED_FIXES,
        rmse_ap20=np.nan, n_ap20=0, ap20_coverage=0.0,
    )
    if out["anchor_ok"]:
        weights = 1.0 / np.clip(cov[trusted], 1e-4, None)
        yaw, tx, ty = weighted_rigid_fit_2d(src_xy[trusted], dst_xy[trusted], weights)
        out["yaw"], out["tx"], out["ty"] = yaw, tx, ty
        out["rmse_cpt7"] = rigid_rmse_2d(src_xy[trusted], dst_xy[trusted], yaw, tx, ty)

    ap20 = fit_dlio_to_ap20(extracted_dir, ros1_dir)
    if ap20:
        out["rmse_ap20"], out["n_ap20"], out["ap20_coverage"] = (
            ap20["rmse"], ap20["n"], ap20["coverage"],
        )

    return out
