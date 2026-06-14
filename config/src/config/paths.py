"""Central filesystem paths, overridable via environment variables."""

import os
from pathlib import Path

DATASET_ROOT = Path(
    os.environ.get("GRAND_TOUR_DATASET_ROOT", "/datahome/lmertens/grand_tour_dataset")
)
ROS1_ROOT = DATASET_ROOT / "ros1"  # raw bags
OUTPUT_ROOT = DATASET_ROOT / "output"  # extraction + labeling output

WAYFASTER_DATASET_ROOT = Path(
    os.environ.get("WAYFASTER_DATASET_ROOT", "/datahome/lmertens/wayfaster_dataset")
)
