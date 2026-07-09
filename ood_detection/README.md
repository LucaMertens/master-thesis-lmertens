# ood_detection

Post-hoc OOD detection pipeline for WayFASTER/GrandTour checkpoints (RQ2/RQ3/optional
RQ4-RQ6). Two steps, run from the repo root:

```bash
# 1. Extract cached features/scores for every mission tier (id_train, id_heldout,
#    near_forest, near_gravel, far_snow, far_arc). ~10-20 min per tier on one GPU.
uv run --project ood_detection python -m ood_detection.features.extract \
    --cfg training/traversability_network/configs/grandtour.yaml \
    --checkpoint /datahome/lmertens/checkpoints/hxf3z45m/best_epoch=33-step=32334.ckpt \
    --out-dir /datahome/lmertens/ood_cache/hxf3z45m \
    --device cuda:0

# 2. AUROC/FPR95 table + plots across every detector x tier combination.
uv run --project ood_detection python -m ood_detection.eval.run \
    --features-dir /datahome/lmertens/ood_cache/hxf3z45m --plots
```

For DINOv3 instead of the default `dinov2_vitb14_reg`, pass
`--dino-model dinov3_<variant> --dino-weights /path/to/local.pth` (gated weights
must be downloaded manually; see the facebookresearch/dinov3 hub repo).

For a checkpoint trained with `MODEL.EVIDENTIAL: True`
(`configs/grandtour_evidential.yaml`), step 2 automatically adds the `evidential`
detector; nothing else changes.

## Layout

- `common/` — checkpoint loading, DINO loading, tier/mission definitions, window metadata
- `features/` — step 1, one script
- `mahalanobis/` — post-hoc feature-distance detector (Mahalanobis++, Ren et al. 2021),
  used on both the WayFASTER penultimate feature and a DINO feature (optional RQ4)
- `evidential/` — Deep Evidential Regression OOD score (the head/loss itself lives in
  `training/traversability_network`, not here)
- `ensemble/` — optional RQ5, RGB+D vs RGB-only prediction disagreement
- `temporal/` — optional RQ6, EMA smoothing of per-window scores
- `eval/` — step 2, AUROC/FPR95 + plots

See `agents/rq3_failure_modes.md` for the failure-mode analysis plan built on top
of this pipeline's cached outputs.
