# training

Trains the traversability network (WayFASTER adapted to GrandTour) on RHE labels and relative human labels (`adaptations/human_pairs.py`, `adaptations/rizz.py`). Code is in `training/traversability_network/`; all changes to upstream code live in `src/traversability_network/adaptations/`.

- `train.py`: entry point; configs in `training/traversability_network/configs/`.
- `run_eval_grid.sh`: runs a fixed list of configs, one lane per GPU.
- Checkpoints go to `<CKPT_BASE>/<wandb_id>/`; `--resume` continues a config's last wandb run.
- `adaptations/visualize_predictions.py`: projects predicted mu and nu onto camera images for a qualitative check.

## Usage

```bash
uv run python -m traversability_network.train --cfg_file training/traversability_network/configs/grandtour_v2.yaml
```
