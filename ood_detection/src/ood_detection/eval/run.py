"""Step 2 of the OOD pipeline: read the cached feature .npz files (from
ood_detection.features.extract) and produce an AUROC/FPR95 table across every
detector x tier combination, plus score-histogram/ROC plots.

Detectors:
  mahalanobis_wayfaster / mahalanobis_dino  - post-hoc feature distance (RQ2),
      identical detector class on two feature spaces (answers optional RQ4 too)
  modality_disagreement                     - optional RQ5, cheap pseudo-ensemble
  evidential                                - RQ2, only if the checkpoint has a
      Deep Evidential Regression head (MODEL.EVIDENTIAL)
Each detector is also scored with a temporal EMA applied to its raw scores
(optional RQ6), added as "<detector>+ema" rows.

Tiers: near_forest, near_gravel, far_snow, far_arc (id_heldout is negative class
for all of them), plus sensor_dropout (id_heldout scored under simulated depth
dropout, using the *_rgbonly feature/score columns as the "OOD" condition).

--cell-maps additionally scores any {tier}_cellmaps.npz (written by features.extract
with --max-cell-maps) per grid cell with the already-fit mahalanobis_wayfaster
detector, for RQ3 spatial failure-mode plots: where in the BEV grid a window's
OOD score actually comes from, rather than one pooled number per window.
"""

import argparse
import os

import numpy as np
import pandas as pd

from ood_detection.ensemble.modality_disagreement import disagreement_score
from ood_detection.evidential.score import evidential_score, has_evidential
from ood_detection.eval.metrics import evaluate
from ood_detection.eval.plots import plot_roc, plot_score_hist
from ood_detection.mahalanobis.detector import MahalanobisDetector
from ood_detection.temporal.ema import ema_by_mission

OOD_TIERS = ["near_forest", "near_gravel", "far_snow", "far_arc"]


def load_tier(features_dir, tier):
    path = os.path.join(features_dir, f"{tier}.npz")
    return dict(np.load(path, allow_pickle=True)) if os.path.exists(path) else None


def load_cellmaps(features_dir, tier):
    path = os.path.join(features_dir, f"{tier}_cellmaps.npz")
    return dict(np.load(path, allow_pickle=True)) if os.path.exists(path) else None


def save_cell_score_maps(features_dir, out_dir, detector, tiers):
    maps_dir = os.path.join(out_dir, "cell_maps")
    for tier in ["id_heldout"] + tiers:
        data = load_cellmaps(features_dir, tier)
        if data is None:
            continue
        os.makedirs(maps_dir, exist_ok=True)
        n, x, y, c = data["maps"].shape
        scores = detector.score(data["maps"].reshape(-1, c).astype(np.float32)).reshape(n, x, y)
        for i in range(n):
            name = f"{tier}_{data['mission'][i]}_{data['timestamp'][i]:.2f}"
            np.save(os.path.join(maps_dir, f"{name}.npy"), scores[i])
        print(f"{tier}: {n} cell score maps -> {maps_dir}")


def build_detectors(id_train):
    detectors = {
        "mahalanobis_wayfaster": MahalanobisDetector().fit(id_train["bev_feat"]),
        "mahalanobis_dino": MahalanobisDetector().fit(id_train["dino_feat"]),
    }
    feat_keys = {"mahalanobis_wayfaster": "bev_feat", "mahalanobis_dino": "dino_feat"}
    return detectors, feat_keys


def score(name, data, detectors, feat_keys, rgbonly=False):
    if name in detectors:
        key = feat_keys[name] + ("_rgbonly" if rgbonly else "")
        return detectors[name].score(data[key])
    if name == "modality_disagreement":
        return disagreement_score(data)
    if name == "evidential":
        return evidential_score(data, rgbonly=rgbonly)
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", required=True, help="output dir of ood_detection.features.extract")
    ap.add_argument("--out-dir", default=None, help="default: <features-dir>/eval")
    ap.add_argument("--ema-halflife", type=float, default=1.0, help="seconds")
    ap.add_argument("--plots", action="store_true")
    ap.add_argument("--cell-maps", action="store_true",
                     help="score {tier}_cellmaps.npz per grid cell (needs --max-cell-maps at extraction)")
    args = ap.parse_args()
    out_dir = args.out_dir or os.path.join(args.features_dir, "eval")
    os.makedirs(out_dir, exist_ok=True)

    id_train = load_tier(args.features_dir, "id_train")
    id_heldout = load_tier(args.features_dir, "id_heldout")
    assert id_train is not None and id_heldout is not None, "need id_train.npz and id_heldout.npz"

    detectors, feat_keys = build_detectors(id_train)
    names = ["mahalanobis_wayfaster", "mahalanobis_dino", "modality_disagreement"]
    if has_evidential(id_train):
        names.append("evidential")
    else:
        print("no evidential head in this checkpoint (MODEL.EVIDENTIAL off); skipping evidential detector")

    rows = []
    for name in names:
        s_id = score(name, id_heldout, detectors, feat_keys)

        for tier in OOD_TIERS:
            data = load_tier(args.features_dir, tier)
            if data is None:
                continue
            s_ood = score(name, data, detectors, feat_keys)
            rows.append({"detector": name, "tier": tier, **evaluate(s_id, s_ood)})
            if args.plots:
                tag = f"{name}_{tier}"
                plot_score_hist(s_id, s_ood, tag, os.path.join(out_dir, f"hist_{tag}.png"))
                plot_roc(s_id, s_ood, tag, os.path.join(out_dir, f"roc_{tag}.png"))

            s_id_ema = ema_by_mission(s_id, id_heldout["mission"], id_heldout["timestamp"], args.ema_halflife)
            s_ood_ema = ema_by_mission(s_ood, data["mission"], data["timestamp"], args.ema_halflife)
            rows.append({"detector": f"{name}+ema", "tier": tier, **evaluate(s_id_ema, s_ood_ema)})

        # Sensor dropout: id_heldout scored under simulated depth loss (the *_rgbonly
        # columns) as the "OOD" condition against the same RGB+D-fitted detector.
        # modality_disagreement is skipped here: it *is* the depth-off signal itself.
        if name != "modality_disagreement":
            s_dropout = score(name, id_heldout, detectors, feat_keys, rgbonly=True)
            rows.append({"detector": name, "tier": "sensor_dropout", **evaluate(s_id, s_dropout)})

    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "results.csv")
    df.to_csv(csv_path, index=False)
    print(df.to_string(index=False))
    print(f"\nresults -> {csv_path}")

    if args.cell_maps:
        save_cell_score_maps(args.features_dir, out_dir, detectors["mahalanobis_wayfaster"], OOD_TIERS)


if __name__ == "__main__":
    main()
