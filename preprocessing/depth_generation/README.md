# depth_generation

Runs [FoundationStereo](https://github.com/NVlabs/FoundationStereo) on the Alphasense
stereo pairs to produce depth maps for each mission. Lives in Docker because
FoundationStereo needs CUDA 12.4 / torch 2.4; separate from the uv workspace.

## Setup

**Model weights** — download the `23-51-11` folder from the
[FoundationStereo Google Drive](https://drive.google.com/drive/folders/1VhPebc_mMxWKccrv7pdQLTvXYVcLYpsf)
(or run `./download_model.sh`) and put it at `depth_generation/pretrained_models/23-51-11/`
(needs `model_best_bp2.pth` and `cfg.yaml`). The Dockerfile bakes this folder into
the image (a bind mount would not work: the docker daemon is the host's and the
devcontainer's repo path does not exist there), so it must exist before the first build.

## Run

Don't invoke `run.sh` directly — it's the Docker launcher used internally by
`preprocessing/pipeline.py --depth` (needs a Docker + GPU host; not this
devcontainer). Run the other extraction steps first, then:

```bash
uv run python preprocessing/pipeline.py --mission ETH-1 ETH-2 --depth --gpus '"device=0,1"'
```

`run.sh` (re)builds the image — cheap when nothing changed; even the first build only
downloads wheels, nothing compiles from source — then mounts
`/datahome/lmertens/grand_tour_dataset/output/` at `/data`. Depth is written back
next to each mission's images.

All requested missions are processed by a single container. `generate_depth.py`
spawns one worker per visible GPU (override with `--num-gpus`); each worker takes
the next mission off a shared queue and runs it end-to-end on its own device, so
up to `--num-gpus` missions are depth-processed in parallel.

## Output

Each mission gets:
- `depth/<ts_ns>.png` — 16-bit PNG, millimetres, 0 = invalid
- `depth_manifest.npz` — `{ts_ns, path}` index

Depth is in the `front_center` frustum (aligned with the RGB images), 1440×1080.
