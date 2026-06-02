# depth_generation

Runs [FoundationStereo](https://github.com/NVlabs/FoundationStereo) on the Alphasense
stereo pairs to produce depth maps for each mission. Lives in Docker because
FoundationStereo needs CUDA 12.4 / Python 3.11; separate from the uv workspace.

## Setup

**Model weights** — download the `23-51-11` folder from the
[FoundationStereo Google Drive](https://drive.google.com/drive/folders/1VhPebc_mMxWKccrv7pdQLTvXYVcLYpsf)
and put it at `depth_generation/pretrained_models/23-51-11/`
(needs `model_best_bp2.pth` and `cfg.yaml`).

**Build the image** (takes a while — flash-attn compiles from source):
```bash
cd depth_generation
./build.sh
```

## Run

Inputs come from `uv run python -m grandtour_extraction.preprocess` (run that first).

```bash
cd depth_generation

# all GPUs
./run.sh --missions /data/ETH-1

# specific GPU
GPUS='"device=0"' ./run.sh --missions /data/ETH-1

# multiple missions
GPUS='"device=0"' ./run.sh --missions /data/ETH-1 /data/ETH-3
```

`run.sh` mounts `/datahome/lmertens/grand_tour_dataset/output/` at `/data` and the
weights read-only. Depth is written back next to each mission's images.

Override defaults:
```bash
DATA_DIR=/abs/path/to/output MODELS_DIR=/abs/path/to/weights GPUS='"device=1"' \
  ./run.sh --missions /data/ETH-1
```

## Output

Each mission gets:
- `depth/<ts_ns>.png` — 16-bit PNG, millimetres, 0 = invalid
- `depth_manifest.npz` — `{ts_ns, path}` index

Depth is in the `front_center` frustum (aligned with the RGB images), 1440×1080.
