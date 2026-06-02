#!/usr/bin/env bash
# Build the depth-generation image.
#   ./build.sh                 # builds grand-tour-depth:latest
#   IMAGE=foo:dev ./build.sh   # custom tag
# Extra args are forwarded to `docker build` (e.g. --no-cache, --build-arg MAX_JOBS=8).
set -euo pipefail

IMAGE="${IMAGE:-grand-tour-depth:latest}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec docker build -t "${IMAGE}" "$@" "${SCRIPT_DIR}"
