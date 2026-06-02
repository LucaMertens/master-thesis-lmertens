"""Extract camera extrinsics + CPT7 odometry per mission.

Writes cam2base.npy (base <- rectified front_center), stereo_extrinsics.npz (baseline),
and cpt7_odom.npy (N x 8: t, x, y, z, qw, qx, qy, qz in ENU).
"""

import argparse
import pathlib
from typing import Any

import numpy as np
from data_labeling_pipeline.ros1_transforms import get_static_transform
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore
from scipy.spatial.transform import Rotation as R

from . import rectify

ODOM_TOPIC = "/boxi/inertial_explorer/tc/odometry"


def _rotate_frame(t_base_cam: np.ndarray, r_raw_rect: np.ndarray) -> np.ndarray:
    """base <- rectified-cam, given base <- raw-cam and (raw -> rect) rotation."""
    out = t_base_cam.copy()
    out[:3, :3] = t_base_cam[:3, :3] @ r_raw_rect.T
    return out


def extract_static(mission_dir: pathlib.Path, balance: float, fov_scale: float) -> dict[str, np.ndarray]:
    rect = rectify.build_rectifiers(mission_dir, balance, fov_scale)
    cam2base = _rotate_frame(rect.t_base_ref, rect.R_ref)  # base <- rectified front_center
    geom_baseline = float(np.linalg.norm(rect.t_base_ref[:3, 3] - rect.t_base_partner[:3, 3]))
    print(f"baseline: {rect.baseline:.4f} m (from P2) vs {geom_baseline:.4f} m (tf centers)")
    return {"cam2base": cam2base, "baseline": np.array(rect.baseline)}


def extract_cpt7_odom(mission_dir: pathlib.Path) -> np.ndarray:
    r_base_cpt7 = get_static_transform(mission_dir, "base", "cpt7_imu").rotation
    bags = list(mission_dir.glob("*cpt7_ie_tc*.bag"))
    if not bags:
        raise FileNotFoundError(f"No *cpt7_ie_tc*.bag in {mission_dir}")

    typestore = get_typestore(Stores.ROS1_NOETIC)
    rows = []
    for bag in bags:
        with Reader(bag) as reader:
            conns = [c for c in reader.connections if c.topic == ODOM_TOPIC]
            for conn, _, raw in reader.messages(conns):
                msg: Any = typestore.deserialize_ros1(raw, conn.msgtype)
                t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                p = msg.pose.pose.position
                q = msg.pose.pose.orientation
                r_enu_base = R.from_quat([q.x, q.y, q.z, q.w]) * r_base_cpt7.inv()
                qx, qy, qz, qw = r_enu_base.as_quat()
                rows.append([t, p.x, p.y, p.z, qw, qx, qy, qz])

    if not rows:
        raise ValueError(f"No messages on {ODOM_TOPIC} in {mission_dir}")
    odom = np.asarray(rows, np.float64)
    return odom[np.argsort(odom[:, 0])]


def extract(mission_dir: pathlib.Path, output_dir: pathlib.Path,
            balance: float, fov_scale: float) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    static = extract_static(mission_dir, balance, fov_scale)
    np.save(output_dir / "cam2base.npy", static["cam2base"])
    np.savez(output_dir / "stereo_extrinsics.npz", baseline=static["baseline"])
    odom = extract_cpt7_odom(mission_dir)
    np.save(output_dir / "cpt7_odom.npy", odom)
    print(f"cpt7_odom: {odom.shape} samples, t in [{odom[0,0]:.2f}, {odom[-1,0]:.2f}]")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mission-dir", required=True, type=pathlib.Path)
    p.add_argument("--output-dir", required=True, type=pathlib.Path)
    p.add_argument("--balance", type=float, default=rectify.DEFAULT_BALANCE)
    p.add_argument("--fov-scale", type=float, default=rectify.DEFAULT_FOV_SCALE)
    args = p.parse_args()
    extract(args.mission_dir, args.output_dir, args.balance, args.fov_scale)


if __name__ == "__main__":
    main()
