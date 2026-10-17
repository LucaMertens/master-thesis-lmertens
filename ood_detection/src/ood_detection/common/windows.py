"""Per-window metadata not exposed by GrandTourDataset.__getitem__."""

import bisect

import numpy as np
from scipy.spatial.transform import Rotation as R


def locate(ds, idx):
    """(rosbag dict, index of the window's anchor frame within it)."""
    phys_idx = ds.indices[idx]
    bag_idx = bisect.bisect_right(ds.bag_sizes, phys_idx) - 1
    rosbag = ds.rosbags[bag_idx]
    return rosbag, phys_idx - ds.bag_sizes[bag_idx] + ds.n_frames - 1


def anchor_timestamp(ds, idx):
    """Timestamp of a window's last input frame (the anchor)."""
    rosbag, i = locate(ds, idx)
    return float(rosbag["image_timestamp"][i])


def anchor_pose(ds, idx):
    """(x, y, yaw) of the window's BEV frame in the mission odometry frame (same state get_states anchors on)."""
    rosbag, i = locate(ds, idx)
    ts = rosbag["states_timestamp"]
    j = min(bisect.bisect_left(ts, rosbag["image_timestamp"][i]), len(ts) - 1)
    pos = np.asarray(rosbag["position"])[j]
    yaw = R.from_quat(np.asarray(rosbag["quaternion"])[j]).as_euler("ZYX")[0]
    return np.array([pos[0], pos[1], yaw], dtype=np.float64)
