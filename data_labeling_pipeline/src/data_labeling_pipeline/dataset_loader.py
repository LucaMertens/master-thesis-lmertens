import pathlib

import numpy as np
import zarr
from scipy.spatial.transform import Rotation

from .zarr_transforms import get_static_transform

TIME_RANGE_WARNING_THRESHOLD = 1.0  # seconds


def get_shared_timestamps(tstamps1, tstamps2, dt):
    # Timestamps are in unix seconds.
    start1, end1, start2, end2 = tstamps1[0], tstamps1[-1], tstamps2[0], tstamps2[-1]
    # Check if the difference between time-ranges is too large
    start_difference = abs(start1 - start2)
    end_difference = abs(end1 - end2)
    if (
        start_difference > TIME_RANGE_WARNING_THRESHOLD
        or end_difference > TIME_RANGE_WARNING_THRESHOLD
    ):
        print(
            "Warning: Time ranges of datasets do not sufficiently overlap. "
            f"Start time difference: {start_difference:.2f}s ({start1} vs {start2}), End time difference: {end_difference:.2f}s ({end1} vs {end2})"
        )

    start_time = max(start1, start2)
    end_time = min(end1, end2)

    return np.arange(start_time, end_time, dt)


def quaternions_to_yaws(quaternions, scalar_first=False):
    """Convert quaternions stored as [w, x, y, z] (scalar_first) or [x, y, z, w] (scalar_last) o yaw angles (degrees)."""
    quaternions = np.asarray(quaternions)
    if quaternions.ndim != 2 or quaternions.shape[1] != 4:
        raise ValueError("Quaternion array must have shape (N, 4).")

    print("Using scalar_first =", scalar_first, "for quaternion to yaw conversion.")

    rotation = Rotation.from_quat(quaternions, scalar_first=scalar_first)

    return rotation.as_euler("zyx", degrees=False)[:, 0]


def interpolate_unwrapped_angles(timestamps, angles, shared_timestamps):
    """Interpolate angles after unwrapping to avoid discontinuities at +/- pi."""
    return np.interp(shared_timestamps, timestamps, np.unwrap(angles))


def fill_command_gaps_with_zero_commands(
    timestamps: np.ndarray,
    linear: np.ndarray,
    angular: np.ndarray,
    gap_factor: float = 4.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sometimes, there are gaps in the data where no measurements were recorded for a while. Fill them with a const value."""
    timestamps = np.asarray(timestamps)
    linear = np.asarray(linear)
    angular = np.asarray(angular)

    if timestamps.size < 2:
        return timestamps, linear, angular

    diffs = np.diff(timestamps)
    if np.any(diffs < 0):
        raise ValueError("Timestamps must be sorted in ascending order.")

    nominal_dt = float(np.median(diffs))
    amount_of_regular_diffs = np.isclose(diffs, nominal_dt, rtol=0.05).sum()
    print(
        f"Median dt: {nominal_dt:.3f}s, regular diffs: {amount_of_regular_diffs}/{len(diffs)}"
    )
    max_gap = gap_factor * nominal_dt

    if np.all(a=diffs <= max_gap):
        return timestamps, linear, angular

    gap_indices = np.flatnonzero(diffs > max_gap)
    zero_linear = np.zeros((1, linear.shape[1]), dtype=linear.dtype)
    zero_angular = np.zeros((1, angular.shape[1]), dtype=angular.dtype)

    timestamp_parts = []
    linear_parts = []
    angular_parts = []

    start_index = 0
    for gap_index in gap_indices:
        next_index = gap_index + 1

        timestamp_parts.append(timestamps[start_index:next_index])
        linear_parts.append(linear[start_index:next_index])
        angular_parts.append(angular[start_index:next_index])

        missing_ts = np.arange(
            timestamps[gap_index] + nominal_dt,
            timestamps[next_index],
            nominal_dt,
            dtype=timestamps.dtype,
        )
        if missing_ts.size:
            timestamp_parts.append(missing_ts)
            linear_parts.append(np.repeat(zero_linear, missing_ts.size, axis=0))
            angular_parts.append(np.repeat(zero_angular, missing_ts.size, axis=0))

        start_index = next_index

    timestamp_parts.append(timestamps[start_index:])
    linear_parts.append(linear[start_index:])
    angular_parts.append(angular[start_index:])

    return (
        np.concatenate(timestamp_parts).astype(timestamps.dtype, copy=False),
        np.concatenate(linear_parts).astype(linear.dtype, copy=False),
        np.concatenate(angular_parts).astype(angular.dtype, copy=False),
    )


class GrandTourLoader:
    def __init__(self, mission_path, scalar_first_quaternions=False):
        self.mission_path = pathlib.Path(mission_path)
        self.data_path = self.mission_path / "data"
        # We are not entirely sure whether quaternions are stored as (w, x, y, z) or (x, y, z, w), but most likely scalar_last (x, y, z, w).
        self.scalar_first_quaternions = scalar_first_quaternions

    def load_topic(self, topic_name):
        """Loads a given zarr topic group and returns its arrays."""
        topic_path = self.data_path / topic_name
        if not topic_path.exists():
            raise FileNotFoundError(f"Topic not found: {topic_path}")
        return zarr.open(str(topic_path), mode="r")

    def load_subtopic(self, topic_name, subtopic_name):
        """Loads a leaf subtopic array from a topic group."""
        topic = self.load_topic(topic_name)
        if subtopic_name not in topic:
            raise KeyError(f"Subtopic not found: {topic_name}/{subtopic_name}")
        return np.asarray(topic[subtopic_name])

    def load_commands(self, fill_gaps=True):
        # Load command data (U = [v_k, omega_k]). Frame: body
        cmd_ts = np.asarray(self.load_subtopic("anymal_command_twist", "timestamp"))
        cmd_linear = self.load_subtopic("anymal_command_twist", "linear")
        cmd_angular = self.load_subtopic("anymal_command_twist", "angular")

        if fill_gaps:
            cmd_ts, cmd_linear, cmd_angular = fill_command_gaps_with_zero_commands(
                cmd_ts, cmd_linear, cmd_angular
            )
        return cmd_ts, cmd_linear, cmd_angular

    def load_odometry(self):
        # Load odometry data. Frame: ENU-origin (world-ish, origin is where the robot started)
        odom_ts = np.asarray(self.load_subtopic("cpt7_ie_rt_odometry", "timestamp"))
        odom_pos = self.load_subtopic("cpt7_ie_rt_odometry", "pose_pos")
        odom_orien = self.load_subtopic("cpt7_ie_rt_odometry", "pose_orien")
        return odom_ts, odom_pos, odom_orien

    def get_cpt7_imu_to_base_transform(self):
        return get_static_transform(self.load_topic("."), "cpt7_imu", "base")

    def map_cpt7_imu_pose_to_base(self, positions, orientations):
        """Map dynamic poses from ENU->cpt7_imu to ENU->base using static TF."""
        t_cpt7_imu_to_base = self.get_cpt7_imu_to_base_transform()
        r_cpt7_imu_to_base = Rotation.from_matrix(t_cpt7_imu_to_base[:3, :3])
        t_cpt7_imu_to_base_vec = t_cpt7_imu_to_base[:3, 3]

        r_enu_to_cpt7_imu = Rotation.from_quat(
            orientations, scalar_first=self.scalar_first_quaternions
        )
        r_enu_to_base = r_enu_to_cpt7_imu * r_cpt7_imu_to_base
        p_enu_to_base = positions + r_enu_to_cpt7_imu.apply(t_cpt7_imu_to_base_vec)

        return p_enu_to_base, r_enu_to_base

    def load_data_shared_timestamps(self, dt=0.1, fill_gaps=True):
        """
        Loads topics and returns them with shared timestamps, resampled at a uniform rate of dt seconds.

        Returns:
        - T: 1D array of uniformly spaced, shared timestamps
        - U: 2D array of control inputs (v_k, omega_k) at each timestamp
        - Z: 3D array of states (px, py, yaw) at each timestamp
        """
        cmd_ts, cmd_linear, cmd_angular = self.load_commands(fill_gaps=fill_gaps)
        v_k = cmd_linear[:, 0]
        omega_k = cmd_angular[:, 2]

        odom_ts, odom_pos, odom_orien = self.load_odometry()
        odom_pos_base, odom_rot_base = self.map_cpt7_imu_pose_to_base(
            odom_pos, odom_orien
        )
        px = odom_pos_base[:, 0]
        py = odom_pos_base[:, 1]
        odom_yaw = odom_rot_base.as_euler("zyx", degrees=False)[:, 0]

        shared_ts = get_shared_timestamps(cmd_ts, odom_ts, dt=dt)

        # Interpolate commands and states to shared timestamps
        v_k_shared = np.interp(shared_ts, cmd_ts, v_k)
        omega_k_shared = np.interp(shared_ts, cmd_ts, omega_k)
        px_shared = np.interp(shared_ts, odom_ts, px)
        py_shared = np.interp(shared_ts, odom_ts, py)
        odom_yaw_shared = interpolate_unwrapped_angles(odom_ts, odom_yaw, shared_ts)

        U = np.column_stack((v_k_shared, omega_k_shared))
        Z = np.column_stack((px_shared, py_shared, odom_yaw_shared))

        return shared_ts, U, Z
