"""Central filesystem paths, overridable via environment variables."""

import os
from pathlib import Path

DATASET_ROOT = Path(
    os.environ.get("GRAND_TOUR_DATASET_ROOT", "/datahome/lmertens/grand_tour_dataset")
)
ROS1_ROOT = DATASET_ROOT / "ros1"  # raw bags
EXTRACTED_ROOT = DATASET_ROOT / "extracted"  # per-mission sensor extraction (images, depth, odom, ...)
LABELS_ROOT = DATASET_ROOT / "labels"  # per-mission labeling pipeline output (rhe/, simple/)
OUTPUT_ROOT = DATASET_ROOT / "output"  # graphs, plots, analysis results

WAYFASTER_DATASET_ROOT = Path(
    os.environ.get("WAYFASTER_DATASET_ROOT", "/datahome/lmertens/wayfaster_dataset")
)
