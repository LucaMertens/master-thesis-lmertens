"""ForEnt dataset: the GrandTour reader with Go2 odometry, D435i depth range and entrapment spans."""

import copy
import os

import numpy as np
from torch.utils.data import ConcatDataset

from traversability_network.adaptations.grandtour_dataset import GrandTourDataset

import per_dataset_affine
from paths import FORENT_EXTRACTED_ROOT, FORENT_LABELS_ROOT

# D435i returns usable depth well past its 3 m "ideal" range (7-13% of valid pixels are at
# 5-8 m). Matching dbound keeps both datasets supported over the same BEV volume.
DEPTH_TRUST_RANGE = 8.0


class ForEntDataset(GrandTourDataset):
    ODOM_FILE = "odom.npy"
    CAMERA_KEY = "front_center"
    DEPTH_MAX_RANGE = DEPTH_TRUST_RANGE
    DATASET_ID = per_dataset_affine.FORENT_ID

    def read_mission(self, mission_dir, rhe_dir):
        segments = super().read_mission(mission_dir, rhe_dir)
        spans = self.read_entrapments(mission_dir)
        for segment in segments:
            segment["entrapment"] = spans
        return segments

    @staticmethod
    def read_entrapments(mission_dir):
        """(N, 4) hand-labelled spans: start, end, type id, severity."""
        path = os.path.join(mission_dir, "entrapment.npy")
        return np.load(path) if os.path.exists(path) else np.zeros((0, 4))


def forent_configs(configs):
    """Copy of ``configs`` with DATASET roots on ForEnt (no resized/depth caches)."""
    out = copy.deepcopy(configs)
    out.DATASET.DATA_ROOT = str(FORENT_EXTRACTED_ROOT)
    out.DATASET.LABELS_ROOT = str(FORENT_LABELS_ROOT)
    out.DATASET.RESIZED_ROOT = None
    out.DATASET.DEPTH_CACHE_ROOT = None
    return out


def build_train_dataset(configs, grandtour_dataset, transform=None):
    """GrandTour train set plus the configured ForEnt sequences, reusing GrandTour LDS weights."""
    sequences = list(configs.DATASET.FORENT_SEQUENCES)
    if not sequences:
        return grandtour_dataset
    forent = ForEntDataset(
        forent_configs(configs),
        sequences,
        transform=transform,
        weights=grandtour_dataset.weights,
        train=True,
    )
    repeat = int(configs.DATASET.FORENT_REPEAT)
    print(
        f"Mixed train set: {len(grandtour_dataset)} GrandTour + "
        f"{repeat} x {len(forent)} ForEnt windows ({len(sequences)} sequences)"
    )
    return ConcatDataset([grandtour_dataset] + [forent] * repeat)
