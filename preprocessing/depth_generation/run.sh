#!/usr/bin/env bash
# Run depth generation in the container.
#
# GPU selection mirrors .devcontainer/devcontainer.json's `--gpus` runArg.
# Examples:
#   GPUS=all ./run.sh --missions /data/ETH-1
#   GPUS='"device=0"' ./run.sh --missions /data/ETH-1          # single GPU
#   GPUS='"device=0,2"' ./run.sh --missions /data/ETH-1        # expose two GPUs
#   ./run.sh --missions /data/ETH-1 --stride 5                 # process every 5th image
#
# NOTE: generate_depth.py uses a single CUDA device (device 0 of whatever the
# container sees). Exposing several GPUs does not parallelise the run; it just
# selects which physical device is used.
#
# Override host paths via env:
#   DATA_DIR    host dir containing per-mission folders (mounted at /data)
#   MODELS_DIR  host dir with pretrained_models/23-51-11/ (mounted read-only)
#   IMAGE       image tag (default: grand-tour-depth:latest)
set -euo pipefail

GPUS="${GPUS:-all}"
IMAGE="${IMAGE:-grand-tour-depth:latest}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATA_DIR="${DATA_DIR:-/datahome/lmertens/grand_tour_dataset/output}"
# MODELS_DIR="${MODELS_DIR:-${SCRIPT_DIR}/pretrained_models}"
MODELS_DIR="${MODELS_DIR:-/home/lmertens/git-repos/master-thesis-lmertens/preprocessing/depth_generation/pretrained_models}"

if [ "$#" -eq 0 ]; then
    set -- --missions /data/ETH-1
fi

exec docker run --rm \
    --gpus "${GPUS}" \
    --shm-size=32gb \
    -v "${DATA_DIR}:/data" \
    -v "${MODELS_DIR}:/app/pretrained_models:ro" \
    "${IMAGE}" \
    --ckpt /app/pretrained_models/23-51-11/model_best_bp2.pth \
    "$@"
