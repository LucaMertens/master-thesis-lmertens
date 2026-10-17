"""Rescores cached windows under the segment tiers of common.retier.

Distance detectors (metric x feature space, median cell distance) and reference-free ones
(disagreement, depth entropy, predicted mu). Also sweeps the cell aggregator and reports
per-mission ID typicality.
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
import torch

from ood_detection.common.retier import PENDING, TIERS, assign
from ood_detection.common.viewpoint import disagreement, disagreement_spread
from ood_detection.ensemble.modality_disagreement import disagreement_score
from ood_detection.eval.fingerprint import beats_null, pairwise_null_aurocs
from ood_detection.eval.metrics import auroc, evaluate
from ood_detection.eval.run import full_windows, load_tier
from ood_detection.mahalanobis.detector import MahalanobisPP

SPACES = {"bev": "cells", "dino": "dino_feat", "image": "img_cells"}
METRICS = ["mahalanobis", "euclidean", "cosine"]
# Reported aggregator; the sweep shows results are flat over the choice
REPORTED = ("quantile", 0.5)
# Quantiles form one monotone axis; the mean is the extra alternative
AGG_SPECS = ([("quantile", q) for q in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                                        0.9, 0.95, 0.99)] + [("mean", 1.0)])
STACKED = ["mission", "timestamp", "n_cells", "cells", "dino_feat", "mu_map",
           "observed_map", "pose", "mu_pred", "mu_pred_rgbonly", "depth_entropy"]
GEOMETRY = ["centers", "xbound", "ybound"]
VIEWPOINT_K = 10
VIEWPOINT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_all(features_dir):
    """Every cached window, sorted by (mission, timestamp) and deduplicated.

    The npz files are a bag of windows, retiered by (mission, timestamp). Sorting is needed
    for the viewpoint score; stale tier files can hold the same window, hence the dedup."""
    paths = sorted(p for p in glob.glob(os.path.join(features_dir, "*.npz"))
                   if not p.endswith(("_cellmaps.npz", "_imgfeat.npz")))
    parts = [dict(np.load(p, allow_pickle=True)) for p in paths]
    keys = [k for k in STACKED if all(k in p for p in parts)]
    data = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    order = np.lexsort((data["timestamp"], data["mission"]))
    data = {k: v[order] for k, v in data.items()}
    keep = np.ones(len(order), bool)
    keep[1:] = (data["mission"][1:] != data["mission"][:-1]) | (
        data["timestamp"][1:] != data["timestamp"][:-1])
    data = {k: v[keep] for k, v in data.items()}
    data.update({k: parts[0][k] for k in GEOMETRY})
    return attach_sidecar(data, features_dir)


def window_key(mission, timestamp):
    return np.array([f"{m}@{t:.6f}" for m, t in zip(mission, timestamp)])


def attach_sidecar(data, features_dir, suffix="_imgfeat.npz", key="img_cells"):
    """Joins the per-pixel cache of features.extract_image by (mission, timestamp); uncovered windows are flagged unusable."""
    paths = sorted(glob.glob(os.path.join(features_dir, f"*{suffix}")))
    if not paths:
        return data
    parts = [np.load(p) for p in paths]
    lut = {k: i for i, k in enumerate(
        np.concatenate([window_key(p["mission"], p["timestamp"]) for p in parts]))}
    block = np.concatenate([p[key] for p in parts])
    idx = np.array([lut.get(k, -1) for k in window_key(data["mission"], data["timestamp"])])
    out = np.zeros((len(idx), *block.shape[1:]), block.dtype)
    out[idx >= 0] = block[idx[idx >= 0]]
    data[key] = out
    data[f"{key}_ok"] = idx >= 0
    print(f"{key}: {(idx >= 0).sum()}/{len(idx)} windows joined from {len(paths)} sidecars")
    return data


def usable_windows(data, key):
    """Windows whose cached block for this feature space is not padding."""
    if key == "cells":
        return full_windows(data, key)
    if f"{key}_ok" in data:
        return data[f"{key}_ok"]
    return np.ones(len(data["timestamp"]), bool)


def cell_distances(det, block):
    """(N, C) per-cell distance."""
    n, c = block.shape[:2]
    return det.cell_scores(block.reshape(n * c, -1)).reshape(n, c)


def aggregate(d, kind, q):
    """Window score from (N, C) cell distances: quantile or mean."""
    return d.mean(1) if kind == "mean" else np.quantile(d, q, axis=1)


def reference_free(data):
    """Detectors that need no in-distribution fit."""
    out = {}
    bounds = {"xbound": data["xbound"], "ybound": data["ybound"]}
    mu = data["mu_map"].astype(np.float32)
    args = (mu, data["observed_map"], data["pose"], data["mission"], data["centers"], bounds)
    out["viewpoint_disagreement"] = disagreement(*args, VIEWPOINT_K,
                                                 device=VIEWPOINT_DEVICE)
    spread = disagreement_spread(*args, device=VIEWPOINT_DEVICE)
    out["viewpoint_range"] = spread[:, 0]
    out["viewpoint_range_med"] = spread[:, 1]
    out["viewpoint_variance"] = spread[:, 2]
    out["viewpoint_variance_med"] = spread[:, 3]
    out["modality_disagreement"] = disagreement_score(data)
    out["predicted_trav"] = -data["mu_pred"]
    if "depth_entropy" in data and not np.all(np.isnan(data["depth_entropy"])):
        out["depth_entropy"] = data["depth_entropy"]
    return out


def loo_scores(data, tier, id_tiers=("id_train", "id_heldout"), **kw):
    """Leave-one-mission-out distance scores over ID windows (others stay NaN).

    Fitting on id_train alone would make train missions look typical by construction."""
    in_id = np.isin(tier, id_tiers)
    spaces = {s: k for s, k in SPACES.items() if k in data}
    out = {f"{s}_{m}": np.full(len(tier), np.nan) for s in spaces for m in METRICS}
    for space, key in spaces.items():
        block = data[key].astype(np.float32)
        if block.ndim == 2:
            block = block[:, None, :]
        usable = usable_windows(data, key)
        for held in np.unique(data["mission"][in_id]):
            fit = in_id & usable & (data["mission"] != held)
            sel = np.flatnonzero(in_id & usable & (data["mission"] == held))
            for metric in METRICS:
                det = MahalanobisPP(metric=metric, **kw).fit(
                    block[fit].reshape(-1, block.shape[-1]))
                out[f"{space}_{metric}"][sel] = aggregate(
                    cell_distances(det, block[sel]), *REPORTED)
    return out


def measure(name, s, tier, data, train_missions, rows, typicality, raws, typ_all,
            tiers=TIERS):
    """Appends per-tier results, medians and typicality rows.

    The null is the folded pairwise AUROC between ID recordings (session floor)."""
    mission = data["mission"]
    ref = s[tier == "id_heldout"]
    in_id = np.isin(tier, ["id_train", "id_heldout"]) & np.isfinite(s)
    null = pairwise_null_aurocs(s[in_id], mission[in_id])
    for t in tiers:
        if t not in ("id_train", "id_heldout") and (tier == t).any():
            sel = (tier == t) & np.isfinite(s)
            per_mission = [auroc(ref[np.isfinite(ref)], s[sel & (mission == m)])
                           for m in np.unique(mission[sel])]
            rows.append({"detector": name, "tier": t, **evaluate(ref, s[tier == t]),
                         "null_mean": float(null.mean()),
                         "beats_null": beats_null(per_mission, null)})
    for t in tiers:
        v = s[(tier == t) & np.isfinite(s)]
        if len(v):
            raws.append({"detector": name, "tier": t, "p50": float(np.median(v))})
    for m in np.unique(data["mission"][in_id]):
        sel = in_id & (data["mission"] == m)
        typicality.append({"detector": name, "mission": m,
                           "split": "train" if m in train_missions else "heldout",
                           "n": int(sel.sum()), "auroc_vs_id": auroc(s[in_id & ~sel], s[sel])})
    # same typicality reading for every tier mission
    for t in tiers:
        for m in np.unique(mission[(tier == t) & np.isfinite(s)]):
            sel = (tier == t) & (mission == m) & np.isfinite(s)
            typ_all.append({"detector": name, "mission": m, "tier": t, "n": int(sel.sum()),
                            "auroc_vs_id": auroc(s[in_id & ~sel], s[sel])})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", required=True)
    ap.add_argument("--modes", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data = load_all(args.features_dir)
    train_missions = set(np.unique(load_tier(args.features_dir, "id_train")["mission"]))
    tier = assign(data["mission"], data["timestamp"], train_missions)

    counts = pd.Series(tier).replace("", "dropped").value_counts()
    print(f"{len(tier)} windows, {counts.get('dropped', 0)} unassigned "
          f"(pending segment times: {', '.join(PENDING)})")
    print(counts.to_string(), "\n")

    is_train = tier == "id_train"
    rows, typicality, raws, sweep, typ_all = [], [], [], [], []
    for space, key in [(s, k) for s, k in SPACES.items() if k in data]:
        block = data[key].astype(np.float32)
        if block.ndim == 2:
            block = block[:, None, :]
        # partly filled cell blocks are padded
        usable = usable_windows(data, key)
        for metric in METRICS:
            det = MahalanobisPP(args.modes, metric=metric).fit(
                block[is_train & usable].reshape(-1, block.shape[-1]))
            d = cell_distances(det, block)
            for kind, q in AGG_SPECS:
                s = np.where(usable, aggregate(d, kind, q), np.nan)
                ref = s[tier == "id_heldout"]
                for t in TIERS:
                    if t not in ("id_train", "id_heldout") and (tier == t).any():
                        sweep.append({"space": space, "metric": metric, "kind": kind,
                                      "q": q, "tier": t, **evaluate(ref, s[tier == t])})
                if (kind, q) == REPORTED:
                    measure(f"{space}_{metric}", s, tier, data, train_missions,
                            rows, typicality, raws, typ_all, TIERS)

    for name, s in reference_free(data).items():
        measure(name, np.asarray(s, dtype=float), tier, data, train_missions,
                rows, typicality, raws, typ_all, TIERS)

    df, typ, raw = pd.DataFrame(rows), pd.DataFrame(typicality), pd.DataFrame(raws)
    pd.set_option("display.width", 250)
    for metric in ["auroc", "fpr95", "aupr"]:
        print(f"\n== {metric} ==")
        print(df.pivot_table(index="tier", columns="detector", values=metric)
              .reindex([t for t in TIERS if t in set(df["tier"])]).round(3).to_string())

    print("\n== aggregator sweep, AUROC averaged over tiers ==")
    print(pd.DataFrame(sweep).pivot_table(index=["space", "metric"], columns=["kind", "q"],
                                          values="auroc").round(3).to_string())

    print("\n== ID typicality (AUROC of one ID mission vs the rest of ID) ==")
    print(typ.pivot_table(index=["split", "mission"], columns="detector",
                          values="auroc_vs_id").round(3).to_string())

    out = args.out or os.path.join(args.features_dir, "eval")
    os.makedirs(out, exist_ok=True)
    df.to_csv(os.path.join(out, "retiered_results.csv"), index=False)
    typ.to_csv(os.path.join(out, "retiered_typicality.csv"), index=False)
    pd.DataFrame(typ_all).to_csv(os.path.join(out, "retiered_typicality_all.csv"), index=False)
    raw.to_csv(os.path.join(out, "retiered_raw.csv"), index=False)
    pd.DataFrame(sweep).to_csv(os.path.join(out, "retiered_aggregator.csv"), index=False)
    print(f"\n-> {out}/retiered_results.csv")


if __name__ == "__main__":
    main()
