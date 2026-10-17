#!/usr/bin/env bash

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG_DIR="training/traversability_network/configs"
LOG_DIR="logs"
TRAIN_PATTERN=" -m traversability_network\.train"

LANE_0=(gt_backbone_lr_10percent_f2 gt_weight_decay_0.01_f2 gt_depth_weight_0.5_f2)
LANE_1=(gt_loss_gaussian_binned_f2 gt_weight_decay_0.1_f2 gt_wayfaster_pretrained_f2)
LANE_2=(gt_loss_focal_regression_f2 gt_decoder_slim_f2 gt_wayfaster_pretrained_frozen_f2)
LANE_3=(gt_backbone_frozen_f2 gt_decoder_minimal_frozen_f2)

if pgrep -af "$TRAIN_PATTERN" | grep -qvE "pgrep|/bin/bash -c|run_eval_grid"; then
    echo "training already running, refusing to start the grid"
    exit 1
fi

run_lane() {
    local gpu="$1"; shift
    for tag in "$@"; do
        echo "$(date '+%F %T') gpu $gpu starting $tag"
        CUDA_VISIBLE_DEVICES="$gpu" uv run python -u -m traversability_network.train \
            --cfg_file "$CONFIG_DIR/$tag.yaml" --resume > "$LOG_DIR/$tag.log" 2>&1
        echo "$(date '+%F %T') gpu $gpu finished $tag (exit $?)"
        sleep 10
    done
}

mkdir -p "$LOG_DIR"
run_lane 0 "${LANE_0[@]}" &
run_lane 1 "${LANE_1[@]}" &
run_lane 2 "${LANE_2[@]}" &
run_lane 3 "${LANE_3[@]}" &
wait
echo "$(date '+%F %T') grid done"
