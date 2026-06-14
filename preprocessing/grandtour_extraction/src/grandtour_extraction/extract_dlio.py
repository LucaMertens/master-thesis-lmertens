"""Extract DLIO lidar-map odometry per mission.

Writes dlio_odom.npy (N x 8: t, x, y, z, qx, qy, qz, qw), the dlio_map->base
pose. dlio_map is gravity-aligned (DLIO inits from IMU gravity), so z is up as
in cpt7's ENU; the yaw origin is the mission's initial heading.
"""

import argparse
import pathlib
from typing import Any

import numpy as np
from data_labeling_pipeline.pipeline_step import is_step_done, log_step, write_step_done
from data_labeling_pipeline.ros1_transforms import get_static_transform
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore

ODOM_TOPIC = "/boxi/dlio/lidar_map_odometry"

# Bump to invalidate dlio_odom.npy from older versions of this step.
DLIO_VERSION = 3


def extract_dlio_odom(mission_dir: pathlib.Path) -> np.ndarray:
    bags = list(mission_dir.glob("*dlio*.bag"))
    if not bags:
        raise FileNotFoundError(f"No *dlio*.bag in {mission_dir}")

    typestore = get_typestore(Stores.ROS1_NOETIC)
    rows, child_frame = [], None
    for bag in bags:
        with Reader(bag) as reader:
            conns = [c for c in reader.connections if c.topic == ODOM_TOPIC]
            for conn, _, raw in reader.messages(conns):
                msg: Any = typestore.deserialize_ros1(raw, conn.msgtype)
                child_frame = msg.child_frame_id
                t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                p = msg.pose.pose.position
                q = msg.pose.pose.orientation
                rows.append([t, p.x, p.y, p.z, q.x, q.y, q.z, q.w])

    if not rows:
        raise ValueError(f"No messages on {ODOM_TOPIC} in {mission_dir}")
    arr = np.asarray(rows, np.float64)
    arr = arr[np.argsort(arr[:, 0])]

    # Full SE3 lidar->base so position picks up the lever arm, matching dataset_loader.
    t_lidar_to_base = get_static_transform(mission_dir, child_frame, "base")
    pos_base, rot_base = t_lidar_to_base.apply_to_poses(arr[:, 1:4], arr[:, 4:8])

    odom = np.empty_like(arr)
    odom[:, 0] = arr[:, 0]
    odom[:, 1:4] = pos_base
    odom[:, 4:8] = rot_base.as_quat()
    return odom


def extract(
    mission_dir: pathlib.Path, output_dir: pathlib.Path, force: bool = False
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not force and is_step_done(output_dir, "dlio", DLIO_VERSION):
        print(f"Skipping dlio (up to date): {output_dir}")
        return

    with log_step("dlio", output_dir):
        if not list(mission_dir.glob("*dlio*.bag")):
            print(f"No *dlio*.bag in {mission_dir}, skipping dlio odometry")
            write_step_done(output_dir, "dlio", DLIO_VERSION, no_data=True)
            return
        odom = extract_dlio_odom(mission_dir)
        np.save(output_dir / "dlio_odom.npy", odom)
        print(
            f"dlio_odom: {odom.shape} samples, t in [{odom[0, 0]:.2f}, {odom[-1, 0]:.2f}]"
        )
        write_step_done(output_dir, "dlio", DLIO_VERSION)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--mission-dir", required=True, type=pathlib.Path)
    p.add_argument("--output-dir", required=True, type=pathlib.Path)
    p.add_argument("--force", action="store_true", help="Re-run even if up to date")
    args = p.parse_args()
    extract(args.mission_dir, args.output_dir, force=args.force)


if __name__ == "__main__":
    main()
