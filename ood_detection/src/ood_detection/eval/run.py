"""Step 2 of the OOD pipeline: scores cached features and reports detection metrics."""

import argparse
import os

import numpy as np
import pandas as pd

from shared_utils.tiers import ID_TRAIN, tiers
from ood_detection.common.splits import HELDOUT_TIER
from ood_detection.common.viewpoint import disagreement
from ood_detection.ensemble.modality_disagreement import disagreement_score
from ood_detection.evidential.score import evidential_score, has_evidential
from ood_detection.eval.grouping import block_bootstrap, time_blocks
from ood_detection.eval.metrics import auroc, auroc_per_mission, evaluate
from ood_detection.eval.plots import plot_roc, plot_score_hist
from ood_detection.mahalanobis.detector import MahalanobisPP
from ood_detection.temporal.ema import ema_by_mission

OOD_TIERS = [t for t in tiers() if t not in (ID_TRAIN, HELDOUT_TIER)]
# Cell-level AUROC only makes sense where the whole scene is shifted
WHOLE_SCENE_TIERS = [t for t in OOD_TIERS if t.startswith("far_")]
CELL_AUROC_MAX = 200_000
VIEWPOINT_K = 10


def load_tier(features_dir, tier, suffix=""):
    path = os.path.join(features_dir, f"{tier}{suffix}.npz")
    return dict(np.load(path, allow_pickle=True)) if os.path.exists(path) else None


def load_cellmaps(features_dir, tier):
    return load_tier(features_dir, tier, "_cellmaps")


def full_windows(data, key="cells"):
    """Windows whose cached cell block holds only observed cells."""
    return data["n_cells"] >= data[key].shape[1]


def cell_rows(data, key="cells"):
    """(N, D) cells of the fully observed windows, one row each."""
    block = data[key][full_windows(data, key)]
    return block.reshape(-1, block.shape[-1]).astype(np.float32)


def save_cell_score_maps(features_dir, out_dir, detector, tiers):
    maps_dir = os.path.join(out_dir, "cell_maps")
    for tier in ["id_heldout"] + tiers:
        data = load_cellmaps(features_dir, tier)
        if data is None:
            continue
        os.makedirs(maps_dir, exist_ok=True)
        n, x, y, c = data["maps"].shape
        scores = detector.cell_scores(data["maps"].reshape(-1, c).astype(np.float32)).reshape(n, x, y)
        scores[~data["observed"]] = np.nan
        for i in range(n):
            name = f"{tier}_{data['mission'][i]}_{data['timestamp'][i]:.2f}"
            np.save(os.path.join(maps_dir, f"{name}.npy"), scores[i])
        print(f"{tier}: {n} cell score maps -> {maps_dir}")


def build_detectors(id_train, n_modes=3, top_frac=0.1):
    """One detector per feature space and distance, fitted on ID training. kmeans_* drops the shared covariance."""
    X = {"wayfaster": cell_rows(id_train), "dino": id_train["dino_feat"]}
    detectors, feat_keys = {}, {}
    for space, feats in X.items():
        for tag, metric in (("mahalanobis", "mahalanobis"), ("kmeans", "euclidean")):
            name = f"{tag}_{space}"
            detectors[name] = MahalanobisPP(n_modes, top_frac, metric=metric).fit(feats)
            feat_keys[name] = "cells" if space == "wayfaster" else "dino_feat"
    return detectors, feat_keys


def viewpoint_score(data, k=VIEWPOINT_K, top_frac=0.1):
    """|delta mu| between a window and the window k steps later, on the ground both see."""
    bounds = {"xbound": data["xbound"], "ybound": data["ybound"]}
    return disagreement(data["mu_map"].astype(np.float32), data["observed_map"],
                        data["pose"], data["mission"], data["centers"], bounds, k, top_frac)


def score(name, data, detectors, feat_keys, rgbonly=False):
    if name in detectors:
        key = feat_keys[name] + ("_rgbonly" if rgbonly else "")
        block = data[key].astype(np.float32)
        if block.ndim == 2:  # one pooled vector per window
            block = block[:, None, :]
        s = detectors[name].frame_scores(block)
        return np.where(full_windows(data, key), s, np.nan) if key.startswith("cells") else s
    if name == "predicted_trav":
        # baseline: lower predicted traversability = more OOD
        return -data["mu_pred_rgbonly" if rgbonly else "mu_pred"]
    if name == "viewpoint_disagreement":
        return viewpoint_score(data)
    if name == "modality_disagreement":
        return disagreement_score(data)
    if name == "evidential":
        return evidential_score(data, rgbonly=rgbonly)
    raise ValueError(name)


def auroc_ci(s_id, id_data, s_ood, ood_data, n_boot):
    """95% CI over 30 s time blocks: consecutive windows are not independent."""
    s = np.concatenate([s_id, s_ood])
    label = np.concatenate([np.zeros(len(s_id), bool), np.ones(len(s_ood), bool)])
    mission = np.concatenate([id_data["mission"], ood_data["mission"]])
    timestamp = np.concatenate([id_data["timestamp"], ood_data["timestamp"]])
    keep = np.isfinite(s)
    blocks = time_blocks(mission[keep], timestamp[keep])
    s, label = s[keep], label[keep]

    def metric(idx):
        y = label[idx]
        return auroc(s[idx][~y], s[idx][y]) if y.any() and not y.all() else np.nan

    return block_bootstrap(metric, blocks, n_boot)


def cell_auroc(detector, id_data, ood_data, key="cells", seed=0):
    """Per-cell AUROC, ID cells against tier cells."""
    rng = np.random.default_rng(seed)
    scores = []
    for data in (id_data, ood_data):
        X = cell_rows(data, key)
        if len(X) > CELL_AUROC_MAX:
            X = X[rng.choice(len(X), CELL_AUROC_MAX, replace=False)]
        scores.append(detector.cell_scores(X))
    return auroc(*scores)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", required=True, help="output dir of ood_detection.features.extract")
    ap.add_argument("--out-dir", default=None, help="default: <features-dir>/eval")
    ap.add_argument("--ema-halflife", type=float, default=1.0, help="seconds")
    ap.add_argument("--modes", type=int, nargs="+", default=[3],
                    help="k-means modes; the first is the headline detector, the rest a sensitivity table")
    ap.add_argument("--top-frac", type=float, default=0.1, help="worst share of a window's cells")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--drop-id-missions", nargs="+", default=[],
                    help="held-out ID missions to exclude from the negative class")
    ap.add_argument("--plots", action="store_true")
    ap.add_argument("--cell-maps", action="store_true",
                     help="score {tier}_cellmaps.npz per grid cell (needs --max-cell-maps at extraction)")
    args = ap.parse_args()
    out_dir = args.out_dir or os.path.join(args.features_dir, "eval")
    os.makedirs(out_dir, exist_ok=True)

    id_train = load_tier(args.features_dir, "id_train")
    id_heldout = load_tier(args.features_dir, "id_heldout")
    assert id_train is not None and id_heldout is not None, "need id_train.npz and id_heldout.npz"
    if args.drop_id_missions:
        keep = ~np.isin(id_heldout["mission"], args.drop_id_missions)
        id_heldout = {k: v[keep] for k, v in id_heldout.items()}
        print(f"held-out ID without {args.drop_id_missions}: {keep.sum()} windows")

    detectors, feat_keys = build_detectors(id_train, args.modes[0], args.top_frac)
    names = ["mahalanobis_wayfaster", "mahalanobis_dino", "kmeans_wayfaster", "kmeans_dino",
             "viewpoint_disagreement", "modality_disagreement", "predicted_trav"]
    if has_evidential(id_train):
        names.append("evidential")
    else:
        print("no evidential head in this checkpoint (MODEL.EVIDENTIAL off); skipping evidential detector")

    rows, cell_rows_out = [], []
    for name in names:
        s_id = score(name, id_heldout, detectors, feat_keys)

        for tier in OOD_TIERS:
            data = load_tier(args.features_dir, tier)
            if data is None:
                continue
            s_ood = score(name, data, detectors, feat_keys)
            lo, hi = auroc_ci(s_id, id_heldout, s_ood, data, args.n_boot)
            rows.append({"detector": name, "tier": tier, **evaluate(s_id, s_ood),
                         "auroc_mission": auroc_per_mission(s_id, s_ood, data["mission"]),
                         "auroc_lo": lo, "auroc_hi": hi})
            if args.plots:
                tag = f"{name}_{tier}"
                plot_score_hist(s_id, s_ood, tag, os.path.join(out_dir, f"hist_{tag}.png"))
                plot_roc(s_id, s_ood, tag, os.path.join(out_dir, f"roc_{tag}.png"))

            s_id_ema = ema_by_mission(s_id, id_heldout["mission"], id_heldout["timestamp"], args.ema_halflife)
            s_ood_ema = ema_by_mission(s_ood, data["mission"], data["timestamp"], args.ema_halflife)
            rows.append({"detector": f"{name}+ema", "tier": tier, **evaluate(s_id_ema, s_ood_ema)})

            if name == "mahalanobis_wayfaster" and tier in WHOLE_SCENE_TIERS:
                cell_rows_out.append({"detector": name, "tier": tier,
                                      "cell_auroc": cell_auroc(detectors[name], id_heldout, data)})

        # sensor dropout: id_heldout with simulated depth loss (*_rgbonly) as the OOD condition
        if name not in ("modality_disagreement", "viewpoint_disagreement",
                        "mahalanobis_dino", "kmeans_dino"):  # depth-blind or itself the signal
            s_dropout = score(name, id_heldout, detectors, feat_keys, rgbonly=True)
            rows.append({"detector": name, "tier": "sensor_dropout", **evaluate(s_id, s_dropout)})

    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "results.csv")
    df.to_csv(csv_path, index=False)
    print(df.to_string(index=False))
    print(f"\nresults -> {csv_path}")

    if cell_rows_out:
        cells_df = pd.DataFrame(cell_rows_out)
        cells_df.to_csv(os.path.join(out_dir, "cell_auroc.csv"), index=False)
        print("\n" + cells_df.to_string(index=False))

    if len(args.modes) > 1:
        sweep = []
        tier_data = {t: load_tier(args.features_dir, t) for t in OOD_TIERS}
        for k in args.modes:
            det, keys = build_detectors(id_train, k, args.top_frac)
            for name in det:
                s_id = score(name, id_heldout, det, keys)
                for tier, data in tier_data.items():
                    if data is None:
                        continue
                    s_ood = score(name, data, det, keys)
                    sweep.append({"n_modes": k, "detector": name, "tier": tier,
                                  **evaluate(s_id, s_ood)})
        sweep_df = pd.DataFrame(sweep)
        sweep_df.to_csv(os.path.join(out_dir, "modes_sweep.csv"), index=False)
        print("\n" + sweep_df.to_string(index=False))

    if args.cell_maps:
        save_cell_score_maps(args.features_dir, out_dir, detectors["mahalanobis_wayfaster"], OOD_TIERS)


if __name__ == "__main__":
    main()
