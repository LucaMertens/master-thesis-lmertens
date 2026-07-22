"""Flip conjugation and RHE-derived ground height, absent from the original pipeline."""
import numpy as np

# Horizontal flip mirrors the base frame (y -> -y) and the camera x-axis (the
# image u-flip), so cam2base conjugates as FLIP_BASE @ cam2base @ FLIP_CAM.
FLIP_BASE = np.diag([1.0, -1.0, 1.0, 1.0])
FLIP_CAM = np.diag([-1.0, 1.0, 1.0, 1.0])


def flip_cam2base(cam2base):
    """Full mirror conjugation; negating y translation alone leaves the mount misalignment un-mirrored."""
    return FLIP_BASE @ cam2base @ FLIP_CAM


def ground_height(rosbag_dict, idx, position):
    """Ground z in the anchor frame: base z plus the RHE base-to-ground offset (0 if absent)."""
    if "ground_z" in rosbag_dict:
        ground = np.asarray(rosbag_dict["ground_z"])[idx]
    else:
        ground = np.zeros(len(idx))
    return position[:, 2] + ground
