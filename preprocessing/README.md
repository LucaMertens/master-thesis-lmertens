# preprocessing

Turns raw GrandTour ROS1 bags into the per-mission inputs (images, depth, odometry, commands, mu/nu labels) for the traversability network.

- `pipeline.py`: entry point; runs extraction, depth and labeling per mission and skips up-to-date steps.
- `grandtour_extraction/`, `depth_generation/`, `data_labeling_pipeline/`: the three steps. 
- `wayfaster_dataset/`: analysis scripts for the original WayFASTER dataset (videos, BEV maps, odometry error).

## Usage

```bash
uv run python preprocessing/pipeline.py --mission ETH-1 
```
