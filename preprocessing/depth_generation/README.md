# depth_generation

Runs [FoundationStereo](https://github.com/NVlabs/FoundationStereo) on the rectified Alphasense stereo pairs to produce depth for the network input. Runs in Docker, so it needs a GPU host.

- `run_depth.py` and `run.sh`: launch the container, which runs `generate_depth.py`.
- Output in `<data_root>/extracted/<mission>/depth/`: 16-bit PNG in millimetres per timestamp, 0 is invalid.
- Depth is in the `front_center` frame, aligned with the RGB images.

## Usage

```bash
bash preprocessing/depth_generation/download_model.sh   # weights, once
uv run python preprocessing/depth_generation/run_depth.py --mission ETH-1
```
