# ood_detection

Post-hoc OOD detection on the BEV features of a trained WayFASTER/GrandTour checkpoint. Features are cached once per checkpoint, then each detector is scored against each tier in `shared-utils/src/shared_utils/tiers.json`.

- `features/extract.py` caches a uniform sample of 512 observed BEV cells per window, with predictions, for every tier. `features/extract_image.py` adds the image-space features (encoder output before the frustum lift).
- Detectors: `mahalanobis/` (Mahalanobis++, Mueller & Hein 2025, on BEV, encoder and DINO features), depth-head entropy (cached in `features/extract.py`), `common/viewpoint.py` (disagreement across viewpoints), `ensemble/modality_disagreement.py` (RGB+D vs RGB-only).
- `ensemble/mean_z.py`: equally weighted mean of member z-scores, standardized on ID training windows. `temporal/ema.py`: EMA of window scores. `evidential/`: evidential score, needs a checkpoint trained with `MODEL.EVIDENTIAL`.
- `eval/run.py` computes AUROC / FPR95 / AUPR per detector and tier with 30 s block bootstrap CIs. The first `--modes` value is the headline k, the rest form the ablation.
- `id_train` and `id_heldout` follow the config's `TRAIN_MISSIONS` / `VALID_MISSIONS`, so the ID split matches the checkpoint's fold.
- `tiering/`: browser editor for `tiers.json`, which also drives training (`DATASET.TIER_TIME_RANGES`) and extraction. Missions can be split at a timestamp across tiers.
- `report/`: typst report and plots. Thesis figures are built from the eval CSVs in the companion repo (`figures/ood/`).

## Usage

```bash
uv run python -m ood_detection.features.extract --cfg <cfg> --checkpoint <ckpt> --out-dir <cache_dir>
uv run python -m ood_detection.eval.run --features-dir <cache_dir> --modes 3 1 5 8
```
