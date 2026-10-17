# grandtour_extraction

Extracts per-mission inputs from the raw GrandTour ROS1 bags: rectified Alphasense images, extrinsics, DLIO odometry, commands and ground height.

- One `extract_*.py` module per source (`alphasense`, `dlio`, `commands`, `feet`, `cpt7`), to be called by `preprocessing/pipeline.py`.
- `front_center` and `front_right` are stereo-rectified with `front_center` as reference, so depth and the RGB input share one frame.
- `odometry_eval.ipynb` compares odometry against AP20 ground truth.

## Usage

```bash
uv run python preprocessing/pipeline.py --mission ETH-1 --max-frames 30 
```
