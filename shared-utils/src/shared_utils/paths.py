"""Central filesystem paths, overridable via environment variables."""

import os
import subprocess
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

# Per-user local SSD (bind mount from devcontainer.json) for large derived caches.
DATALOCAL_ROOT = Path(os.environ.get("DATALOCAL_ROOT", "/datalocal/lmertens"))

# Repo root and the thesis figures in the companion repo.
REPO_ROOT = Path(__file__).resolve().parents[3]
FIGURES_ROOT = Path(os.environ.get(
    "FIGURES_ROOT", REPO_ROOT.parent / "master-thesis-lm-private-documentation/figures"))

# Human W-RIZZ pairwise label index (written by side_experiments/wrizz_labeling/export_labels.py).
WRIZZ_INDEX_ROOT = Path(os.environ.get(
    "WRIZZ_INDEX_ROOT", REPO_ROOT / "side_experiments/wrizz_labeling/output/index"))


def ensure_datalocal_root() -> Path:
    """Create DATALOCAL_ROOT if missing, via the host Docker daemon (no plain mkdir inside the devcontainer)."""
    if DATALOCAL_ROOT.is_dir():
        return DATALOCAL_ROOT
    base = DATALOCAL_ROOT.parent
    if not base.is_dir():
        raise FileNotFoundError(
            f"{base} not visible in this container. If this is the devcontainer, "
            f"it needs the {DATALOCAL_ROOT} bind mount from devcontainer.json -- "
            "rebuild the container to pick it up."
        )
    subprocess.run(
        ["docker", "run", "--rm", "-v", f"{base}:/data", "alpine", "sh", "-c",
         f"mkdir -p /data/{DATALOCAL_ROOT.name} && "
         f"chown -R {os.getuid()}:{os.getgid()} /data/{DATALOCAL_ROOT.name}"],
        check=True,
    )
    return DATALOCAL_ROOT
