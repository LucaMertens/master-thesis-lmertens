"""Extract ANYmal twist commands per mission.

Writes command.npy (N x 7: t, vx, vy, vz, wx, wy, wz), t = bag receive time.
"""

import argparse
import pathlib
from typing import Any

import numpy as np
from data_labeling_pipeline.pipeline_step import is_step_done, log_step, write_step_done
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore

COMMAND_TOPIC = "/anymal/twist_command"

# Bump to invalidate older command.npy.
COMMAND_VERSION = 1


def extract_command(mission_dir: pathlib.Path) -> np.ndarray:
    bags = list(mission_dir.glob("*anymal_command.bag"))
    if not bags:
        raise FileNotFoundError(f"No *anymal_command.bag in {mission_dir}")

    typestore = get_typestore(Stores.ROS1_NOETIC)
    rows = []
    for bag in bags:
        with Reader(bag) as reader:
            conns = [c for c in reader.connections if c.topic == COMMAND_TOPIC]
            for conn, timestamp, raw in reader.messages(conns):
                msg: Any = typestore.deserialize_ros1(raw, conn.msgtype)
                t = timestamp * 1e-9
                lin, ang = msg.twist.linear, msg.twist.angular
                rows.append([t, lin.x, lin.y, lin.z, ang.x, ang.y, ang.z])

    if not rows:
        raise ValueError(f"No messages on {COMMAND_TOPIC} in {mission_dir}")
    arr = np.asarray(rows, np.float64)
    return arr[np.argsort(arr[:, 0])]


def extract(
    mission_dir: pathlib.Path, output_dir: pathlib.Path, force: bool = False
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not force and is_step_done(output_dir, "command", COMMAND_VERSION):
        print(f"Skipping command (up to date): {output_dir}")
        return

    with log_step("command", output_dir):
        if not list(mission_dir.glob("*anymal_command.bag")):
            print(f"No *anymal_command.bag in {mission_dir}, skipping command extraction")
            write_step_done(output_dir, "command", COMMAND_VERSION, no_data=True)
            return
        cmd = extract_command(mission_dir)
        np.save(output_dir / "command.npy", cmd)
        print(
            f"command: {cmd.shape} samples, t in [{cmd[0, 0]:.2f}, {cmd[-1, 0]:.2f}]"
        )
        write_step_done(output_dir, "command", COMMAND_VERSION)


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
