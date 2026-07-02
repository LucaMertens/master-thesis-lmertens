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

# Per-user local SSD, bind-mounted by .devcontainer/devcontainer.json. Fast and
# persists across container rebuilds/host moves, unlike /home (container-local
# overlay, lost on rebuild) or /datahome (NFS). Use for large derived caches
# (e.g. resized training images) that don't need to be shared/backed up.
DATALOCAL_ROOT = Path(os.environ.get("DATALOCAL_ROOT", "/datalocal/lmertens"))


def ensure_datalocal_root() -> Path:
    """Create DATALOCAL_ROOT (owned by the current user) if it doesn't exist yet.

    /datalocal itself is host infra (provisioned outside this repo); only the
    per-user subdir may need creating. It isn't bind-mounted whole into the
    devcontainer, so creating it requires going through the host's Docker
    daemon (docker-outside-of-docker) rather than a plain mkdir.
    """
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
