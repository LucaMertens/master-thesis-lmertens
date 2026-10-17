"""ForEnt dataset roots, overridable via FORENT_ROOT."""

import os
from pathlib import Path

# raw/ holds the ROS2 bags; extracted/ and labels/ mirror the GrandTour layout per sequence.
FORENT_ROOT = Path(os.environ.get("FORENT_ROOT", "/datahome/lmertens/forent"))
FORENT_RAW_ROOT = FORENT_ROOT / "ForEnt" / "raw"
FORENT_EXTRACTED_ROOT = FORENT_ROOT / "extracted"
FORENT_LABELS_ROOT = FORENT_ROOT / "labels"
