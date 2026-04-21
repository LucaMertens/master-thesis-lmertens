import pathlib

import numpy as np
import zarr
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R


class GrandTourLoader:
    def __init__(self, mission_path):
        self.mission_path = pathlib.Path(mission_path)
        self.data_path = self.mission_path / "data"

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

    @staticmethod
    def _quaternion_wxyz_to_yaw(quaternions_wxyz):
        """Convert quaternions stored as [w, x, y, z] to yaw angles."""
        quaternions_wxyz = np.asarray(quaternions_wxyz)
        if quaternions_wxyz.ndim != 2 or quaternions_wxyz.shape[1] != 4:
            raise ValueError("Quaternion array must have shape (N, 4).")

        quaternions_xyzw = quaternions_wxyz[:, [1, 2, 3, 0]]
        return R.from_quat(quaternions_xyzw).as_euler("zyx")[:, 0]

    @staticmethod
    def _has_valid_orientation(orientations):
        orientations = np.asarray(orientations)
        return (
            orientations.ndim == 2
            and orientations.shape[1] == 4
            and not np.allclose(orientations, 0.0)
        )

    def load_synchronized_data(self, dt=0.1):
        """
        Loads and synchronizes the specified topics, returning uniformly sampled data matrices.
        - cpt7_ie_rt_odometry for position
        - stim320_imu for heading (or fallback to odometry if raw IMU doesn't contain orientation)
        - anymal_command_twist for control inputs
        """
        cmd_ts = np.asarray(self.load_subtopic("anymal_command_twist", "timestamp"))
        cmd_linear = self.load_subtopic("anymal_command_twist", "linear")
        cmd_angular = self.load_subtopic("anymal_command_twist", "angular")
        v_k = cmd_linear[:, 0]
        omega_k = cmd_angular[:, 2]

        odom_ts = np.asarray(self.load_subtopic("cpt7_ie_rt_odometry", "timestamp"))
        odom_pos = self.load_subtopic("cpt7_ie_rt_odometry", "pose_pos")
        odom_orien = self.load_subtopic("cpt7_ie_rt_odometry", "pose_orien")
        px = odom_pos[:, 0]
        py = odom_pos[:, 1]
        odom_yaw = self._quaternion_wxyz_to_yaw(odom_orien)

        imu_ts = np.asarray(self.load_subtopic("stim320_imu", "timestamp"))
        imu_orien = self.load_subtopic("stim320_imu", "orien")

        if imu_orien is None or not self._has_valid_orientation(imu_orien):
            imu_heading = interp1d(
                odom_ts, odom_yaw, bounds_error=False, fill_value="extrapolate"
            )(imu_ts)
        else:
            imu_heading = self._quaternion_wxyz_to_yaw(imu_orien)

        start_time = max(cmd_ts[0], odom_ts[0], imu_ts[0])
        end_time = min(cmd_ts[-1], odom_ts[-1], imu_ts[-1])

        if start_time > 1e17:
            dt_time = int(dt * 1e9)
        elif start_time > 1e11:
            dt_time = int(dt * 1e6)
        else:
            dt_time = dt

        shared_ts = np.arange(start_time, end_time, dt_time)

        f_vk = interp1d(cmd_ts, v_k)
        f_wk = interp1d(cmd_ts, omega_k)
        f_px = interp1d(odom_ts, px)
        f_py = interp1d(odom_ts, py)
        f_imu_sin = interp1d(imu_ts, np.sin(imu_heading))
        f_imu_cos = interp1d(imu_ts, np.cos(imu_heading))

        U = np.column_stack((f_vk(shared_ts), f_wk(shared_ts)))
        Z_px = f_px(shared_ts)
        Z_py = f_py(shared_ts)
        Z_theta = np.arctan2(f_imu_sin(shared_ts), f_imu_cos(shared_ts))
        Z = np.column_stack((Z_px, Z_py, Z_theta))

        return shared_ts, U, Z
