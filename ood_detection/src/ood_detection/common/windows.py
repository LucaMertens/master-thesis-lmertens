"""Per-window metadata not exposed by GrandTourDataset.__getitem__."""

import bisect


def anchor_timestamp(ds, idx):
    """Timestamp of a window's last input frame (the 'anchor' frame used
    throughout trainer.py, e.g. TrainingModule._project_path_to_cam)."""
    phys_idx = ds.indices[idx]
    bag_idx = bisect.bisect_right(ds.bag_sizes, phys_idx) - 1
    rosbag = ds.rosbags[bag_idx]
    data_idx = phys_idx - ds.bag_sizes[bag_idx]
    return float(rosbag["image_timestamp"][data_idx + ds.n_frames - 1])
