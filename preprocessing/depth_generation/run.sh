#!/usr/bin/env bash
# Run depth generation in the container.
#
# GPU selection mirrors .devcontainer/devcontainer.json's `--gpus` runArg.
# Examples:
#   GPUS=all ./run.sh --missions /data/ETH-1
#   GPUS='"device=0"' ./run.sh --missions /data/ETH-1                    # single GPU
#   GPUS='"device=0,1,2,3"' ./run.sh --missions /data/ETH-1 /data/ETH-2  # 2 missions in parallel, 1 GPU each
#   ./run.sh --missions /data/ETH-1 --stride 5                           # process every 5th image
#
# generate_depth.py spawns one worker per GPU the container sees (override
# with --num-gpus), each pulling the next mission off a shared queue and
# processing it on its own device. With a single mission, it runs on one GPU.
#
# Override host paths via env:
#   DATA_DIR    host dir containing per-mission folders (mounted at /data)
#   IMAGE       image tag (default: grand-tour-depth:latest)
#
# Model weights are baked into the image at /app/pretrained_models/23-51-11/
# (see README setup step); they cannot be bind-mounted because the docker daemon
# is the host's and this repo's path does not exist there. The image is (re)built
# before each run - cheap when nothing changed thanks to layer caching. Forward
# extra build flags via BUILD_ARGS, e.g. BUILD_ARGS='--no-cache' ./run.sh ...
set -euo pipefail

GPUS="${GPUS:-all}"
IMAGE="${IMAGE:-grand-tour-depth:latest}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATA_DIR="${DATA_DIR:-/datahome/lmertens/grand_tour_dataset/output}"
WEIGHTS_DIR="${SCRIPT_DIR}/pretrained_models"

if [ ! -f "${WEIGHTS_DIR}/23-51-11/model_best_bp2.pth" ]; then
    echo "Model weights missing: ${WEIGHTS_DIR}/23-51-11/model_best_bp2.pth (see README)" >&2
    exit 1
fi

if [ "$#" -eq 0 ]; then
    set -- --missions /data/ETH-1
fi

docker build -t "${IMAGE}" ${BUILD_ARGS:-} "${SCRIPT_DIR}"

echo "Running depth generation with image ${IMAGE} on GPUs ${GPUS}..."

# --user keeps depth outputs owned by the invoking user instead of root.
exec docker run --rm \
    --gpus "${GPUS}" \
    --shm-size=32gb \
    --user "$(id -u):$(id -g)" \
    -v "${DATA_DIR}:/data" \
    "${IMAGE}" \
    "$@"
