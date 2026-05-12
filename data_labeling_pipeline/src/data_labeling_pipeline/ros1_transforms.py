import pathlib
from dataclasses import dataclass

import numpy as np
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_types_from_msg, get_typestore
from scipy.spatial.transform import Rotation as R


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
            for connection, timestamp, rawdata in reader.messages():
                if connection.topic == "/tf_static":
                    if "tf2_msgs/msg/TFMessage" not in typestore.types:
                        tf2_msg_def = "geometry_msgs/TransformStamped[] transforms"
                        typestore.register(
                            get_types_from_msg(tf2_msg_def, "tf2_msgs/msg/TFMessage")
                        )
                    msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
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
