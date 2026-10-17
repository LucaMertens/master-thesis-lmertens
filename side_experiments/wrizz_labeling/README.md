# wrizz_labeling

Human pairwise traversability labels after Schreiber et al. 2024: the annotator answers "which of these two points is more traversable" for sparse point pairs.

## Usage

```bash
d=side_experiments/wrizz_labeling
uv run python $d/select_frames.py --split id_heldout && uv run python $d/schedule_pairs.py --split id_heldout
uv run python $d/server.py   # http://localhost:8778
```
