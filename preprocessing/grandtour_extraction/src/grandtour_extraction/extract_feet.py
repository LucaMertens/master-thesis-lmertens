"""Writes ground_z.npy (N x 2: t, z): the gravity-aligned offset from the base
origin down to the ground (~-0.5 m while walking). Feet positions come from
forward kinematics over the TF chain in the tf_minimal bag; the mean of the
two lowest feet approximates the stance height.
"""

import argparse
import pathlib
from collections import defaultdict
from typing import Any

import numpy as np
from data_labeling_pipeline.pipeline_step import is_step_done, log_step, write_step_done
from rosbags.highlevel import AnyReader
from scipy.ndimage import median_filter
from scipy.spatial.transform import Rotation as R

LEGS = ("LF", "RF", "LH", "RH")
# Parent frame of each dynamic joint transform in the TF tree.
JOINT_PARENT = {"HIP": "HAA_drive", "THIGH": "HFE_output", "SHANK": "KFE_drive"}
SMOOTH_WINDOW = 1.0  # seconds, median filter against swing-leg contamination

# Bump to invalidate ground_z.npy from older versions of this step.
GROUND_Z_VERSION = 1


def _mat(quat, trans):
    M = np.eye(4)
    M[:3, :3] = R.from_quat(quat).as_matrix()
    M[:3, 3] = trans
    return M


def _mats(rows):
    """(N, 8) rows of (t, qx, qy, qz, qw, x, y, z) -> (N, 4, 4)."""
    M = np.tile(np.eye(4), (len(rows), 1, 1))
    M[:, :3, :3] = R.from_quat(rows[:, 1:5]).as_matrix()
    M[:, :3, 3] = rows[:, 5:8]
    return M


def read_tf(bag_path: pathlib.Path):
    """Static child -> (parent, 4x4) plus time series for the 12 joint
    transforms and base->odom, each sorted (N, 8: t, quat xyzw, trans xyz)."""
    wanted = {f"{leg}_{link}" for leg in LEGS for link in JOINT_PARENT} | {"odom"}
    static, series = {}, defaultdict(list)
    with AnyReader([bag_path]) as reader:
        conns = [c for c in reader.connections if c.topic in ("/tf", "/tf_static")]
        for conn, _, raw in reader.messages(connections=conns):
            msg: Any = reader.deserialize(raw, conn.msgtype)
            for tr in msg.transforms:
                q, t = tr.transform.rotation, tr.transform.translation
                if conn.topic == "/tf_static":
                    static[tr.child_frame_id] = (
                        tr.header.frame_id,
                        _mat([q.x, q.y, q.z, q.w], [t.x, t.y, t.z]),
                    )
                elif tr.child_frame_id in wanted:
                    stamp = tr.header.stamp.sec + tr.header.stamp.nanosec * 1e-9
                    series[tr.child_frame_id].append(
                        (stamp, q.x, q.y, q.z, q.w, t.x, t.y, t.z)
                    )
    dynamic = {}
    for child, rows in series.items():
        arr = np.asarray(rows)
        dynamic[child] = arr[np.argsort(arr[:, 0])]
    return static, dynamic


def chain_to_base(foot: str, static: dict) -> list:
    """Transform steps from base down to `foot`: 4x4 for static links, the
    child frame name as a placeholder for dynamic joints."""
    steps, child = [], foot
    while child != "base":
        if child in static:
            parent, T = static[child]
            steps.append(T)
        else:
            leg, link = child.split("_")
            parent = f"{leg}_{JOINT_PARENT[link]}"
            steps.append(child)
        child = parent
    return steps[::-1]


def _nearest(arr, ref_t):
    idx = np.searchsorted(arr[:, 0], ref_t)
    idx = np.clip(idx, 1, len(arr) - 1)
    left_closer = (ref_t - arr[idx - 1, 0]) <= (arr[idx, 0] - ref_t)
    return arr[np.where(left_closer, idx - 1, idx)]


def compute_ground_z(static: dict, dynamic: dict) -> np.ndarray:
    ref_t = dynamic["LF_HIP"][:, 0]
    # base->odom holds the odom frame in base coordinates; its inverse rotates
    # feet into the gravity-aligned odom axes so z is true height.
    base_rot = R.from_quat(_nearest(dynamic["odom"], ref_t)[:, 1:5]).inv()
    feet_z = []
    for leg in LEGS:
        M = np.tile(np.eye(4), (len(ref_t), 1, 1))
        for step in chain_to_base(f"{leg}_FOOT", static):
            T = _mats(_nearest(dynamic[step], ref_t)) if isinstance(step, str) else step
            M = M @ T
        feet_z.append(base_rot.apply(M[:, :3, 3])[:, 2])
    feet_z = np.sort(np.stack(feet_z), axis=0)
    ground = feet_z[:2].mean(axis=0)
    rate = (len(ref_t) - 1) / max(ref_t[-1] - ref_t[0], 1e-6)
    k = max(1, int(round(SMOOTH_WINDOW * rate)) | 1)
    return np.column_stack((ref_t, median_filter(ground, size=k, mode="nearest")))


def extract(
    mission_dir: pathlib.Path, output_dir: pathlib.Path, force: bool = False
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not force and is_step_done(output_dir, "ground_z", GROUND_Z_VERSION):
        print(f"Skipping ground_z (up to date): {output_dir}")
        return

    with log_step("ground_z", output_dir):
        bags = list(mission_dir.glob("*tf_minimal*.bag"))
        if not bags:
            print(f"No *tf_minimal*.bag in {mission_dir}, skipping ground_z")
            write_step_done(output_dir, "ground_z", GROUND_Z_VERSION, no_data=True)
            return
        static, dynamic = read_tf(bags[0])
        ground = compute_ground_z(static, dynamic)
        np.save(output_dir / "ground_z.npy", ground)
        print(
            f"ground_z: {ground.shape[0]} samples, t in "
            f"[{ground[0, 0]:.2f}, {ground[-1, 0]:.2f}], "
            f"z median {np.median(ground[:, 1]):.3f}"
        )
        write_step_done(output_dir, "ground_z", GROUND_Z_VERSION)


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
