#!/usr/bin/env bash
# Render depth-overlay videos in the container (no GPU needed).
# Usage: ./overlay.sh --missions /data/ETH-1 [--fps 15 --max-depth 30]
# Env: DATA_DIR (mounted at /data), IMAGE (default grand-tour-depth:latest)
set -euo pipefail

IMAGE="${IMAGE:-grand-tour-depth:latest}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${DATA_DIR:-/datahome/lmertens/grand_tour_dataset/output}"

if [ "$#" -eq 0 ]; then
    set -- --missions /data/ETH-1
fi

# Script mounted read-only so edits skip an image rebuild.
exec docker run --rm \
    -v "${DATA_DIR}:/data" \
    -v "${SCRIPT_DIR}/make_overlay_video.py:/app/make_overlay_video.py:ro" \
    --entrypoint python \
    "${IMAGE}" \
    /app/make_overlay_video.py \
    "$@"
