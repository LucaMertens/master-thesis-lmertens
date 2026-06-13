"""Extract CPT7 odometry per mission.

Writes cpt7_odom.npy (N x 8: t, x, y, z, qw, qx, qy, qz in ENU).
"""

import argparse
import pathlib
from typing import Any

import numpy as np
from data_labeling_pipeline.pipeline_step import is_step_done, log_step, write_step_done
from data_labeling_pipeline.ros1_transforms import get_static_transform
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore
from scipy.spatial.transform import Rotation as R

ODOM_TOPIC = "/boxi/inertial_explorer/tc/odometry"

# Bump to invalidate cpt7_odom.npy from older versions of this step.
CPT7_VERSION = 1


def extract_cpt7_odom(mission_dir: pathlib.Path) -> np.ndarray:
    r_base_cpt7_inv = get_static_transform(mission_dir, "base", "cpt7_imu").rotation.inv()
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
                rows.append([t, p.x, p.y, p.z, q.x, q.y, q.z, q.w])

    if not rows:
        raise ValueError(f"No messages on {ODOM_TOPIC} in {mission_dir}")
    arr = np.asarray(rows, np.float64)
    out_quat = (R.from_quat(arr[:, 4:8]) * r_base_cpt7_inv).as_quat()  # x, y, z, w
    odom = np.empty_like(arr)
    odom[:, :4] = arr[:, :4]
    odom[:, 4] = out_quat[:, 3]
    odom[:, 5:8] = out_quat[:, :3]
    return odom[np.argsort(odom[:, 0])]


def extract(mission_dir: pathlib.Path, output_dir: pathlib.Path, force: bool = False) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not force and is_step_done(output_dir, "cpt7", CPT7_VERSION):
        print(f"Skipping cpt7 (up to date): {output_dir}")
        return

    with log_step("cpt7", output_dir):
        if not list(mission_dir.glob("*cpt7_ie_tc*.bag")):
            print(f"No *cpt7_ie_tc*.bag in {mission_dir}, skipping cpt7 odometry")
            write_step_done(output_dir, "cpt7", CPT7_VERSION, no_data=True)
            return
        odom = extract_cpt7_odom(mission_dir)
        np.save(output_dir / "cpt7_odom.npy", odom)
        print(f"cpt7_odom: {odom.shape} samples, t in [{odom[0,0]:.2f}, {odom[-1,0]:.2f}]")
        write_step_done(output_dir, "cpt7", CPT7_VERSION)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mission-dir", required=True, type=pathlib.Path)
    p.add_argument("--output-dir", required=True, type=pathlib.Path)
    p.add_argument("--force", action="store_true", help="Re-run even if up to date")
    args = p.parse_args()
    extract(args.mission_dir, args.output_dir, force=args.force)


if __name__ == "__main__":
    main()
