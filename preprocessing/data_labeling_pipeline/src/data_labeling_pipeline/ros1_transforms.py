import pathlib
from dataclasses import dataclass
from typing import Any

import numpy as np
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_types_from_msg, get_typestore
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


@dataclass(frozen=True)
class SE3Transform:
    rotation: R
    translation: np.ndarray

    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> "SE3Transform":
        return cls(R.from_matrix(matrix[:3, :3]), matrix[:3, 3].copy())

    def as_matrix(self) -> np.ndarray:
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = self.rotation.as_matrix()
        matrix[:3, 3] = self.translation
        return matrix

    def inverse(self) -> "SE3Transform":
        rot_inv = self.rotation.inv()
        return SE3Transform(rot_inv, -rot_inv.apply(self.translation))

    def apply_to_points(self, points: np.ndarray) -> np.ndarray:
        return self.rotation.apply(points) + self.translation

    def apply_to_poses(
        self, positions: np.ndarray, orientations: np.ndarray
    ) -> tuple[np.ndarray, R]:
        r_parent = R.from_quat(orientations)
        r_out = r_parent * self.rotation
        p_out = positions + r_parent.apply(self.translation)
        return p_out, r_out

    def apply_to_twist(
        self, linear_vel: np.ndarray, angular_vel: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Transform twist from parent frame to child frame.
        v_child = R_child_parent * (v_parent + omega_parent x r_parent_child_in_parent)
        omega_child = R_child_parent * omega_parent
        """
        rot_inv = self.rotation.inv()
        # Angular velocity transformation
        omega_child = rot_inv.apply(angular_vel)
        # Linear velocity transformation
        # self.translation is r_parent_child_in_parent
        v_rel = np.cross(angular_vel, self.translation)
        v_child = rot_inv.apply(linear_vel + v_rel)
        return v_child, omega_child


def inv(T_parent_to_child):
    rot = T_parent_to_child[:3, :3]
    t = T_parent_to_child[:3, 3]
    T_child_to_parent = np.eye(4)
    T_child_to_parent[:3, :3] = rot.T
    T_child_to_parent[:3, 3] = -T_child_to_parent[:3, :3] @ t
    return T_child_to_parent


def pq_to_se3(p, q):
    se3 = np.eye(4, dtype=np.float64)
    if isinstance(p, dict) and isinstance(q, dict):
        pos = np.array([p["x"], p["y"], p["z"]], dtype=np.float64)
        quat = np.array([q["x"], q["y"], q["z"], q["w"]], dtype=np.float64)
    elif hasattr(p, "x"):  # ROS message
        pos = np.array([p.x, p.y, p.z], dtype=np.float64)
        quat = np.array([q.x, q.y, q.z, q.w], dtype=np.float64)
    else:
        pos = np.asarray(p, dtype=np.float64)
        quat = np.asarray(q, dtype=np.float64)

    se3[:3, :3] = R.from_quat(quat).as_matrix()
    se3[:3, 3] = pos
    return se3


def transform_points(points_parent, T_parent_to_child):
    if isinstance(T_parent_to_child, SE3Transform):
        return T_parent_to_child.apply_to_points(points_parent)

    points_homo = np.hstack([points_parent, np.ones((points_parent.shape[0], 1))])
    points_child = (T_parent_to_child @ points_homo.T).T[:, :3]
    return points_child


def get_static_transform(mission_path, parent, child) -> SE3Transform:
    typestore = get_typestore(Stores.ROS1_NOETIC)

    tf_bags = list(pathlib.Path(mission_path).glob("*tf*.bag"))
    if not tf_bags:
        raise FileNotFoundError(f"No bag file containing tf found in {mission_path}.")

    edges = {}  # frame -> list of (child_frame, T_parent_to_child)

    def add_edge(p, c, transform):
        if p not in edges:
            edges[p] = []
        if c not in edges:
            edges[c] = []
        edges[p].append((c, transform))
        edges[c].append((p, inv(transform)))

    for tf_bag in tf_bags:
        with Reader(tf_bag) as reader:
            conns = [c for c in reader.connections if c.topic == "/tf_static"]
            if not conns:
                continue
            for connection, timestamp, rawdata in reader.messages(conns):
                if "tf2_msgs/msg/TFMessage" not in typestore.types:
                    tf2_msg_def = "geometry_msgs/TransformStamped[] transforms"
                    typestore.register(
                        get_types_from_msg(tf2_msg_def, "tf2_msgs/msg/TFMessage")
                    )
                msg: Any = typestore.deserialize_ros1(rawdata, connection.msgtype)
                for tr in msg.transforms:
                    p_frame = tr.header.frame_id.lstrip("/")
                    c_frame = tr.child_frame_id.lstrip("/")
                    T_p_c = pq_to_se3(
                        tr.transform.translation, tr.transform.rotation
                    )
                    add_edge(p_frame, c_frame, T_p_c)

    parent = parent.lstrip("/")
    child = child.lstrip("/")

    if parent == child:
        return SE3Transform(R.identity(), np.zeros(3, dtype=np.float64))

    # BFS to find path from parent to child
    queue = [(parent, np.eye(4))]
    visited = {parent}

    while queue:
        curr, T_accum = queue.pop(0)
        if curr == child:
            return SE3Transform.from_matrix(T_accum)

        for nxt, T_edge in edges.get(curr, []):
            if nxt not in visited:
                visited.add(nxt)
                queue.append((nxt, T_accum @ T_edge))

    raise ValueError(f"Could not find transform from {parent} to {child} in tf_static.")


# TODO: Test this
class TransformInterpolator:
    def __init__(self, times, translations, rotations):
        self.times = times
        self.trans_interp = interp1d(
            times,
            translations,
            axis=0,
            bounds_error=False,
            fill_value=(translations[0], translations[-1]),
        )
        self.rot_interp = Slerp(times, R.from_quat(rotations))

    def __call__(self, t):
        trans = self.trans_interp(t)
        rot = self.rot_interp(t)

        if np.isscalar(t):
            return SE3Transform(rot, trans)
        return [SE3Transform(r, tr) for r, tr in zip(rot, trans)]


def get_transform_series(mission_path, parent, child) -> TransformInterpolator:
    typestore = get_typestore(Stores.ROS1_NOETIC)

    tf_bags = list(pathlib.Path(mission_path).glob("*tf*.bag"))
    if not tf_bags:
        raise FileNotFoundError(f"No bag file containing tf found in {mission_path}.")

    parent = parent.lstrip("/")
    child = child.lstrip("/")

    times = []
    translations = []
    rotations = []

    # Note: For now, we assume a direct parent->child or child->parent edge in the /tf topic
    # A full tree search for dynamic transforms at every timestamp is much more complex
    # and usually not needed for direct sensor-to-base or sensor-to-world transforms.

    invert = False
    found_edge = False

    for tf_bag in tf_bags:
        with Reader(tf_bag) as reader:
            for connection, timestamp, rawdata in reader.messages():
                if connection.topic == "/tf":
                    if "tf2_msgs/msg/TFMessage" not in typestore.types:
                        tf2_msg_def = "geometry_msgs/TransformStamped[] transforms"
                        typestore.register(
                            get_types_from_msg(tf2_msg_def, "tf2_msgs/msg/TFMessage")
                        )
                    msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
                    for tr in msg.transforms:
                        p_frame = tr.header.frame_id.lstrip("/")
                        c_frame = tr.child_frame_id.lstrip("/")

                        if p_frame == parent and c_frame == child:
                            found_edge = True
                            t = tr.header.stamp.sec + tr.header.stamp.nanosec * 1e-9
                            times.append(t)
                            translations.append([
                                tr.transform.translation.x,
                                tr.transform.translation.y,
                                tr.transform.translation.z,
                            ])
                            rotations.append([
                                tr.transform.rotation.x,
                                tr.transform.rotation.y,
                                tr.transform.rotation.z,
                                tr.transform.rotation.w,
                            ])
                        elif p_frame == child and c_frame == parent:
                            found_edge = True
                            invert = True
                            t = tr.header.stamp.sec + tr.header.stamp.nanosec * 1e-9
                            times.append(t)
                            translations.append([
                                tr.transform.translation.x,
                                tr.transform.translation.y,
                                tr.transform.translation.z,
                            ])
                            rotations.append([
                                tr.transform.rotation.x,
                                tr.transform.rotation.y,
                                tr.transform.rotation.z,
                                tr.transform.rotation.w,
                            ])

    if not found_edge:
        raise ValueError(
            f"Could not find direct non-static transform between {parent} and {child} in tf."
        )

    times = np.array(times)
    translations = np.array(translations)
    rotations = np.array(rotations)

    # Sort by time
    sort_idx = np.argsort(times)
    times = times[sort_idx]
    translations = translations[sort_idx]
    rotations = rotations[sort_idx]

    # Handle duplicates by taking the first one
    unique_times, unique_idx = np.unique(times, return_index=True)
    times = unique_times
    translations = translations[unique_idx]
    rotations = rotations[unique_idx]

    if invert:
        rots = R.from_quat(rotations)
        rots_inv = rots.inv()
        translations = -rots_inv.apply(translations)
        rotations = rots_inv.as_quat()

    return TransformInterpolator(times, translations, rotations)
