# grandtour_extraction

Turns raw GrandTour ROS1 bags into the per-mission inputs the depth repo and the
traversability network consume. All images are **fisheye-rectified** (Alphasense
front cameras use the `equidistant` model).

## What it produces

Into `/datahome/lmertens/grand_tour_dataset/output/<mission>/`:

- `images/front_{center,right}/<ts_ns>.png` — rectified frames
- `intrinsics.npy`, `intrinsics_front_{center,right}.npy` — rectified pinhole K
- `image_manifest.npz` — per-camera `{ts_ns, path}`
- `cam2base.npy` (base ← rectified `front_center`), `stereo_extrinsics.npz` (`baseline`)
- `dlio_odom.npy` — base pose in the gravity-aligned dlio_map frame (ground-truth odometry)
- `rhe_timestamp.npy`, `rhe_states.npy`, `rhe_trav_labels.npy`,
  `rhe_mu_observability.npy`, `rhe_nu_observability.npy`, `rhe_v_means.npy`,
  `rhe_omega_means.npy`, `rhe_hyperparameters.json`
  — RHE traversability labels (see `data_labeling_pipeline`)

`front_center` (colour) and `front_right` (mono) are stereo-rectified together with
`front_center` as the reference, so FoundationStereo depth lands directly in the
`front_center` frame — `front_center` is the network's RGB. Depth itself is produced
by the external repo and dropped back into the same folder.

## Run

`uv run python preprocessing/pipeline.py` is the single entry point for all
preprocessing steps: extrinsics, images, RHE labels, and (optionally) depth.

```bash
# everything except depth, for every mission found under --bags-root
uv run python preprocessing/pipeline.py

# a single mission
uv run python preprocessing/pipeline.py --mission ETH-1

# quick test on a few frames
uv run python preprocessing/pipeline.py --mission ETH-1 --max-frames 30

# skip images, e.g. when only extrinsics + labels are needed
uv run python preprocessing/pipeline.py --no-images

# only generate depth images, for everything except a few missions
uv run python preprocessing/pipeline.py --exclude ETH-1 ICE-1 --no-images --no-extrinsics --no-labels --depth --gpus '"device=0,1"'

# process 4 missions concurrently (separate processes, threads split across them)
uv run python preprocessing/pipeline.py --parallel-missions 4
```

Each step can be disabled individually with `--no-images`, `--no-extrinsics`,
`--no-labels`. Depth is opt-in via `--depth` (see `depth_generation/README.md`).

Image extraction is CPU-bound (decode/rectify/encode) and uses a thread pool
(`--jobs`, default `min(cpu_count, 32)`). On a many-core machine, running
multiple missions concurrently with `--parallel-missions` can improve overall
throughput; `--jobs` is then split evenly across the parallel missions unless
set explicitly.

Bags are read from `/datahome/lmertens/grand_tour_dataset/ros1/<mission>/`
(override with `--bags-root`). Individual extraction steps are also runnable directly:
`-m grandtour_extraction.extract_alphasense` / `.extract_dlio`.
