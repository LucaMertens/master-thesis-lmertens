#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="$(dirname "${BASH_SOURCE[0]}")/pretrained_models/"

if [ -f "$MODEL_PATH" ]; then
    echo "Model already exists at $MODEL_PATH"
    exit 0
fi

mkdir -p "$(dirname "$MODEL_PATH")"
uv tool run gdown "https://drive.google.com/drive/folders/1BbhoPliFqPJlrtD65TgNX49sJYuYcwA-?usp=drive_link" -O "$MODEL_PATH" --folder
