#!/usr/bin/env bash
# Run depth generation in the container.

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

# --user keeps outputs owned by the invoking user, not root.
exec docker run --rm \
    --gpus "${GPUS}" \
    --shm-size=32gb \
    --user "$(id -u):$(id -g)" \
    -v "${DATA_DIR}:/data" \
    "${IMAGE}" \
    "$@"
