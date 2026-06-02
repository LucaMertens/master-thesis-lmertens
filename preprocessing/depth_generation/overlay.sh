#!/usr/bin/env bash
# Render depth-overlay videos in the container (no GPU needed).
#
# Examples:
#   ./overlay.sh --missions /data/ETH-1
#   ./overlay.sh --missions /data/ETH-1 /data/ETH-3 --fps 15 --max-depth 30
#
# Override host paths via env (same as run.sh):
#   DATA_DIR   host dir containing per-mission folders (mounted at /data)
#   IMAGE      image tag (default: grand-tour-depth:latest)
set -euo pipefail

IMAGE="${IMAGE:-grand-tour-depth:latest}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${DATA_DIR:-/datahome/lmertens/grand_tour_dataset/output}"

if [ "$#" -eq 0 ]; then
    set -- --missions /data/ETH-1
fi

# Mount the script read-only so edits take effect without rebuilding the image.
exec docker run --rm \
    -v "${DATA_DIR}:/data" \
    -v "${SCRIPT_DIR}/make_overlay_video.py:/app/make_overlay_video.py:ro" \
    --entrypoint python \
    "${IMAGE}" \
    /app/make_overlay_video.py \
    "$@"
