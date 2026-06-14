import pathlib
from typing import Any, Iterator

import numpy as np
from config.paths import EXTRACTED_ROOT
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore
from scipy.spatial.transform import Rotation

from .grandtour_missions import get_mission_by_shortname

TIME_RANGE_WARNING_THRESHOLD = 1.0  # seconds


def _is_strictly_monotonic_increasing(arr):
    return np.all(np.diff(arr) > 0)


def get_shared_timestamps(tstamps1, tstamps2, dt):
    assert _is_strictly_monotonic_increasing(tstamps1)
    assert _is_strictly_monotonic_increasing(tstamps2)

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
            "Warning: Time ranges of datasets do not completely overlap. "
            f"Start time difference: {start_difference:.2f}s ({start1} vs {start2}), End time difference: {end_difference:.2f}s ({end1} vs {end2})"
        )

    start_time = max(start1, start2)
    end_time = min(end1, end2)

    return np.arange(start_time, end_time, dt)


def quaternions_to_yaws(quaternions):
    """Convert quaternions stored as [x, y, z, w] to yaw angles (radians)."""
    quaternions = np.asarray(quaternions)
    if quaternions.ndim != 2 or quaternions.shape[1] != 4:
        raise ValueError("Quaternion array must have shape (N, 4).")

    rotation = Rotation.from_quat(quaternions, scalar_first=False)

    return rotation.as_euler("zyx", degrees=False)[:, 0]


def interpolate_unwrapped_angles(timestamps, angles, shared_timestamps):
    """Interpolate angles after unwrapping to avoid discontinuities at +/- pi."""
    return np.interp(shared_timestamps, timestamps, np.unwrap(angles))


class TopicNotFoundError(Exception):
    pass


class MissingExtractedDataError(Exception):
    pass


def _topic_messages(bag_path, topic_name, *, exact_match=True, require_nonempty=True):
    bag_path = pathlib.Path(bag_path)
    with Reader(bag_path) as reader:
        if exact_match:
            connections = [x for x in reader.connections if x.topic == topic_name]
        else:
            connections = [x for x in reader.connections if topic_name in x.topic]

        if not connections:
            available_topics = [x.topic for x in reader.connections]
            raise TopicNotFoundError(
                f"Topic '{topic_name}' not found in bag {bag_path}. "
                f"Available topics: {available_topics}"
            )

        messages: Iterator[tuple[Any, int, bytes]] = reader.messages(connections)

        first = next(messages, None)
        if first is None:
            if require_nonempty:
                raise ValueError(
                    f"No messages found for topic '{topic_name}' in bag {bag_path}"
                )
            return

        yield first
        yield from messages
        # If the returned generator is not fully consumed and kept in memory, the bag file will remain open.


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

        gap_duration = timestamps[next_index] - timestamps[gap_index]
        missing_count = int(np.floor(gap_duration / nominal_dt)) - 1
        if missing_count > 0:
            missing_ts = timestamps[gap_index] + nominal_dt * np.arange(
                1, missing_count + 1, dtype=timestamps.dtype
            )
            missing_ts = missing_ts[missing_ts < timestamps[next_index]]
        else:
            missing_ts = np.empty((0,), dtype=timestamps.dtype)
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
    def __init__(self, mission_path, extracted_dir=None):
        self.mission_path = pathlib.Path(mission_path)
        self.mission_shortname = self.mission_path.name
        self.extracted_dir = (
            pathlib.Path(extracted_dir)
            if extracted_dir is not None
            else EXTRACTED_ROOT / self.mission_shortname
        )
        try:
            mission_info = get_mission_by_shortname(self.mission_shortname)
            self.mission_timestamp = mission_info["timestamp"]
        except ValueError:
            self.mission_timestamp = self.mission_shortname
            print(
                f"Warning: Could not find shortname {self.mission_shortname}, assuming it is a timestamp."
            )

        self.typestore = get_typestore(Stores.ROS1_NOETIC)

    def _load_extracted(self, filename, step_name):
        path = self.extracted_dir / filename
        if not path.exists():
            raise MissingExtractedDataError(
                f"{path} not found. Run the '{step_name}' extraction step first "
                f"(uv run python preprocessing/pipeline.py --mission {self.mission_shortname})."
            )
        return np.load(path)

    def load_commands(self, fill_gaps=True):
        cmd = self._load_extracted("command.npy", "command")
        cmd_ts, cmd_linear, cmd_angular = cmd[:, 0], cmd[:, 1:4], cmd[:, 4:7]

        if fill_gaps:
            cmd_ts, cmd_linear, cmd_angular = fill_command_gaps_with_zero_commands(
                cmd_ts, cmd_linear, cmd_angular
            )
        return cmd_ts, cmd_linear, cmd_angular

    def load_odometry(self):
        odom = self._load_extracted("dlio_odom.npy", "dlio")
        odom_ts, odom_pos, odom_orien = odom[:, 0], odom[:, 1:4], odom[:, 4:8]
        return odom_ts, odom_pos, odom_orien

    def load_data_shared_timestamps(self, dt=0.1, fill_gaps=True, include_lateral_command=False):
        """
        Loads topics and returns them with shared timestamps, resampled at a uniform rate of dt seconds.

        Returns:
        - T: 1D array of uniformly spaced, shared timestamps
        - U: 2D array of control inputs at each timestamp.
             include_lateral_command=False (default): columns [vx_cmd, omega_cmd]
             include_lateral_command=True:            columns [vx_cmd, vy_cmd, omega_cmd]
        - Z: 3D array of states (px, py, yaw) at each timestamp
        """
        cmd_ts, cmd_linear, cmd_angular = self.load_commands(fill_gaps=fill_gaps)
        vx_k = cmd_linear[:, 0]
        vy_k = cmd_linear[:, 1]
        omega_k = cmd_angular[:, 2]

        odom_ts, odom_pos, odom_orien = self.load_odometry()
        px = odom_pos[:, 0]
        py = odom_pos[:, 1]
        odom_yaw = Rotation.from_quat(odom_orien).as_euler("zyx", degrees=False)[:, 0]

        shared_ts = get_shared_timestamps(cmd_ts, odom_ts, dt=dt)
        assert _is_strictly_monotonic_increasing(shared_ts)

        # Interpolate commands and states to shared timestamps
        vx_k_shared = np.interp(shared_ts, cmd_ts, vx_k)
        vy_k_shared = np.interp(shared_ts, cmd_ts, vy_k)
        omega_k_shared = np.interp(shared_ts, cmd_ts, omega_k)
        px_shared = np.interp(shared_ts, odom_ts, px)
        py_shared = np.interp(shared_ts, odom_ts, py)
        odom_yaw_shared = interpolate_unwrapped_angles(odom_ts, odom_yaw, shared_ts)

        if include_lateral_command:
            U = np.column_stack((vx_k_shared, vy_k_shared, omega_k_shared))
        else:
            U = np.column_stack((vx_k_shared, omega_k_shared))
        Z = np.column_stack((px_shared, py_shared, odom_yaw_shared))

        return shared_ts, U, Z

    def load_image_timestamps(self, topic_name="zed2i_right_images"):
        bag_name = f"{self.mission_timestamp}_{topic_name}.bag"
        bag_path = self.mission_path / bag_name
        if not bag_path.exists():
            raise FileNotFoundError(f"Image bag not found: {bag_path}")

        timestamps = []
        # We want any topic that ends with topic_name, e.g. /zed2i/right/image_rect_color/compressed
        messages = _topic_messages(bag_path, topic_name, exact_match=False)
        for connection, timestamp, rawdata in messages:
            ts = timestamp * 1e-9
            timestamps.append(ts)

        return np.array(timestamps)
