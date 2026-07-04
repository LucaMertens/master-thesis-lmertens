# DINO mu-predictability probe

Ceiling test for the traversability network's mean-collapse: if a linear probe on
frozen foundation-model patch features cannot separate low-mu from high-mu terrain
across missions, the mu label is not recoverable from appearance and no training
trick will fix the collapse.

For every low-mu RHE point (and subsampled high-mu negatives) the robot's future
ground contact point is projected into a camera frame 1-3 s before arrival; the
DINO patch feature at that pixel is cached, then evaluated with a
leave-one-mission-out linear probe.

Run from the repo root:

```bash
uv run python side_experiments/dino_mu_probe/extract_features.py   # writes output/features.npz + overlays
uv run python side_experiments/dino_mu_probe/run_probe.py          # per-mission AUROC/AP + PCA plot
```

`extract_features.py --model dinov3_vitb16 --weights <path.pth>` switches to
DINOv3 (weights are license-gated, download manually). Default model is
`dinov2_vitb14_reg` (ungated).
