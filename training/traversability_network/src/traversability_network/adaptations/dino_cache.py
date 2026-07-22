"""Frozen-DINO patch-feature cache written by cache_dino_wayfaster.py, looked up per bag and flip state."""
import os

import numpy as np
import torch

# Per-process handle cache: opened lazily inside each worker, never pickled
# through the DataLoader.
_MEMMAPS = {}


def _memmap(path):
    if path not in _MEMMAPS:
        _MEMMAPS[path] = np.load(path, mmap_mode="r")
    return _MEMMAPS[path]


def attach_dino_cache(data_dict, cache_root, bag_name):
    """Point data_dict at bag_name's cached features (row order matches read_data's images.csv rows)."""
    cache_dir = os.path.join(cache_root, bag_name)
    assert os.path.exists(os.path.join(cache_dir, "features_flip.npy")), (
        f"{bag_name}: DINO cache missing, run cache_dino_wayfaster.py"
    )
    data_dict["dino_feat_file"] = os.path.join(cache_dir, "features.npy")
    data_dict["dino_feat_file_flip"] = os.path.join(cache_dir, "features_flip.npy")
    data_dict["dino_feat_idx"] = np.arange(len(data_dict["image_timestamp"]), dtype=np.int64)


def read_dino_feats(rosbag_dict, data_idx, n_frames, horizontal_flip):
    """Features for one window; the flipped state has its own cache since DINO isn't flip-equivariant."""
    rows = np.asarray(rosbag_dict["dino_feat_idx"][data_idx : data_idx + n_frames])
    key = "dino_feat_file_flip" if horizontal_flip else "dino_feat_file"
    return torch.from_numpy(np.ascontiguousarray(_memmap(rosbag_dict[key])[rows]))
