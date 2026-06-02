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
- `cpt7_odom.npy` — base orientation in ENU (for roll/pitch)

`front_center` (colour) and `front_right` (mono) are stereo-rectified together with
`front_center` as the reference, so FoundationStereo depth lands directly in the
`front_center` frame — `front_center` is the network's RGB. Depth itself is produced
by the external repo and dropped back into the same folder.

## Run

```bash
# one mission (rectified images + extrinsics)
uv run python -m grandtour_extraction.preprocess --mission ETH-1

# quick test on a few frames
uv run python -m grandtour_extraction.preprocess --mission ETH-1 --max-frames 30

# also generate depth (needs a Docker + GPU host; not this devcontainer)
uv run python -m grandtour_extraction.preprocess --mission ETH-1 --depth --gpus '"device=0"'
```

Bags are read from `/datahome/lmertens/grand_tour_dataset/ros1/<mission>/`
(override with `--bags-root`). Individual steps are also runnable:
`-m grandtour_extraction.extract_images` / `.extract_extrinsics` / `.run_depth`.
