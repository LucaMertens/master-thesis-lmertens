

```bash
uv run --project training/traversability_network python -u -m traversability_network.train --cfg_file training/traversability_network/configs/grandtour_lite.yaml 2>&1 | tee train.log
```

```bash
uv run python -m traversability_network.train --cfg_file training/traversability_network/configs/grandtour_v2.yaml  2>&1 | tee train_v2.log
```


```bash
nohup training/run_loss_experiments.sh > training/run_loss_experiments.out 2>&1 &
```

Qualitative check of a checkpoint: predicted mu/nu projected onto the camera image,
next to the same image with the labelled path colored by its true mu. Saves PNGs
for random + low-mu-containing windows from both train and valid missions.

```bash
uv run --project training/traversability_network python -m traversability_network.adaptations.visualize_predictions \
  --cfg_file training/traversability_network/configs/grandtour_v2.yaml \
  --checkpoint /datahome/lmertens/checkpoints/<run_id>/best_*.ckpt \
  --out_dir vis_predictions --device cuda:1