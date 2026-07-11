#!/usr/bin/env bash
# Run the BMC and binned-CE loss experiments back to back. Before each run it waits for any training already in progress

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG_DIR="training/traversability_network/configs"
# pgrep pattern for a live training process (matches the -m module launch and its
# DDP worker subprocesses); the dot is escaped so it is a literal, not any-char.
TRAIN_PATTERN="traversability_network\.train"

wait_for_idle() {
    while pgrep -f "$TRAIN_PATTERN" > /dev/null; do
        echo "$(date '+%F %T') training in progress, waiting 60s..."
        sleep 60
    done
}

run() {
    local cfg="$1" log="$2"
    wait_for_idle
    echo "$(date '+%F %T') starting run: $cfg"
    uv run python -m traversability_network.train --cfg_file "$cfg" 2>&1 | tee "$log"
    echo "$(date '+%F %T') finished run: $cfg (exit ${PIPESTATUS[0]})"
    # Give DDP workers a moment to fully exit before the next wait_for_idle check.
    sleep 10
}

run "$CONFIG_DIR/grandtour_v2_bmc.yaml"    "training/train_v2_bmc.log"
run "$CONFIG_DIR/grandtour_v2_binned.yaml" "training/train_v2_binned.log"

echo "$(date '+%F %T') all loss experiments done"
