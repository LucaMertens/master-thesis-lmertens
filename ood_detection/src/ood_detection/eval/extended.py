"""Deployment-oriented metrics beyond AUROC/FPR95.

uv run python -m ood_detection.eval.extended --features-dir /datahome/lmertens/ood_cache/<run>
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ood_detection.eval.metrics import auroc
from ood_detection.eval.run import OOD_TIERS, build_detectors, load_tier, score
from ood_detection.temporal.ema import ema_by_mission

N_VISIBLE_MIN = 10  # mu_* columns are noisy below this (agents/rq3_failure_modes.md)
# Window-level hazard flag. ood_split.md's per-timestep threshold (0.75) is too
# loose on the 30s-horizon minimum: 45% of ID windows touch it somewhere. 0.6
# flags 11% of ID windows, matching the ~9% frame-level low-mu rate.
LOW_MU = 0.6
COVERAGE = 0.8      # report risk/recall after dropping the 20% most suspicious


def tpr_at_id_quantile(id_scores, ood_scores, q=0.95):
    """Fraction of OOD caught with the threshold set from ID scores alone (the
    deployable mirror image of FPR95, which needs OOD scores to set it)."""
    return float((ood_scores >= np.quantile(id_scores, q)).mean())


def false_alarms_per_minute(scores, mission, timestamp, thresh):
    """Rising threshold crossings per minute of ID driving, counted per mission.
    NaN scores never alarm."""
    events, minutes = 0, 0.0
    for m in np.unique(mission):
        idx = np.flatnonzero(mission == m)
        above = scores[idx[np.argsort(timestamp[idx])]] >= thresh
        events += int((above & ~np.concatenate([[False], above[:-1]])).sum())
        minutes += (timestamp[idx].max() - timestamp[idx].min()) / 60
    return events / minutes


def fingerprint_auroc(scores, mission):
    """Checks whether a detector separates recordings rather than conditions."""
    ms = np.unique(mission)
    pairs = [
        auroc(scores[mission == a], scores[mission == b])
        for i, a in enumerate(ms) for b in ms[i + 1:]
    ]
    return float(np.mean([max(a, 1 - a) for a in pairs]))


def risk_coverage(scores, err):
    """Keep windows from most to least trusted; risk = running mean error of the
    kept set. Returns (coverage, risk, AURC)."""
    e = err[np.argsort(scores)]
    coverage = np.arange(1, len(e) + 1) / len(e)
    risk = np.cumsum(e) / np.arange(1, len(e) + 1)
    return coverage, risk, float(np.trapezoid(risk, coverage))


def hazard_recall_at_coverage(scores, hazard, c=COVERAGE):
    """Of the true low-mu windows, the fraction landing in the dropped (most
    suspicious) 1-c. Random dropping scores 1-c; 1.0 catches every hazard."""
    if not hazard.any():
        return float("nan")
    return float((scores[hazard] > np.quantile(scores, c)).mean())


def risk_block(scores, data):
    """All label-dependent metrics on one window set, gated to n_visible >=
    N_VISIBLE_MIN where the mu_* columns are trustworthy."""
    gate = (data["n_visible"] >= N_VISIBLE_MIN) & np.isfinite(data["mu_err_mean"]) & np.isfinite(scores)
    s, err = scores[gate], data["mu_err_mean"][gate]
    hazard = data["mu_true_min"][gate] <= LOW_MU
    coverage, risk, aurc = risk_coverage(s, err)
    at = np.searchsorted(coverage, COVERAGE)
    return {
        "aurc": aurc,
        "risk_full": float(risk[-1]),
        f"risk@{COVERAGE}": float(risk[at]),
        "spearman": float(spearmanr(s, err).statistic),
        f"hazard_recall@{COVERAGE}": hazard_recall_at_coverage(s, hazard),
        "n": int(gate.sum()),
        "n_hazard": int(hazard.sum()),
    }


def plot_risk_coverage(curves, title, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 4))
    for label, (coverage, risk) in curves.items():
        plt.plot(coverage, risk, lw=1.8, label=label)
    plt.xlabel("coverage (fraction of windows kept)")
    plt.ylabel("risk (mean mu error of kept windows)")
    plt.title(title)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close("all")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", required=True)
    ap.add_argument("--out-dir", default=None, help="default: <features-dir>/eval")
    ap.add_argument("--id-quantile", type=float, default=0.95)
    args = ap.parse_args()
    out_dir = args.out_dir or os.path.join(args.features_dir, "eval")
    os.makedirs(out_dir, exist_ok=True)

    id_train = load_tier(args.features_dir, "id_train")
    id_heldout = load_tier(args.features_dir, "id_heldout")
    tiers = {t: load_tier(args.features_dir, t) for t in OOD_TIERS}
    tiers = {t: d for t, d in tiers.items() if d is not None}
    detectors, feat_keys = build_detectors(id_train)
    names = ["mahalanobis_wayfaster", "mahalanobis_dino", "modality_disagreement"]

    detection_rows, risk_rows = [], []
    for name in names:
        s_id = score(name, id_heldout, detectors, feat_keys)
        finite_id = s_id[np.isfinite(s_id)]
        thresh = np.quantile(finite_id, args.id_quantile)
        fapm = false_alarms_per_minute(s_id, id_heldout["mission"], id_heldout["timestamp"], thresh)
        s_id_ema = ema_by_mission(s_id, id_heldout["mission"], id_heldout["timestamp"])
        fapm_ema = false_alarms_per_minute(
            s_id_ema, id_heldout["mission"], id_heldout["timestamp"],
            np.nanquantile(s_id_ema, args.id_quantile),
        )
        fingerprint = fingerprint_auroc(finite_id, id_heldout["mission"][np.isfinite(s_id)])

        for tier, data in tiers.items():
            s_ood = score(name, data, detectors, feat_keys)
            detection_rows.append({
                "detector": name, "tier": tier,
                f"tpr@id{args.id_quantile:.0%}": tpr_at_id_quantile(finite_id, s_ood[np.isfinite(s_ood)]),
                "false_alarms_per_min_id": fapm,
                "false_alarms_per_min_id_ema": fapm_ema,
                "fingerprint_auroc_id": fingerprint,
            })

        risk_rows.append({"detector": name, "set": "id_heldout", **risk_block(s_id, id_heldout)})
        pooled_scores = np.concatenate([s_id] + [score(name, d, detectors, feat_keys) for d in tiers.values()])
        pooled = {k: np.concatenate([id_heldout[k]] + [d[k] for d in tiers.values()])
                  for k in ("mu_err_mean", "mu_true_min", "n_visible")}
        risk_rows.append({"detector": name, "set": "id+ood", **risk_block(pooled_scores, pooled)})

        curves = {}
        for label, s, d in [("id_heldout", s_id, id_heldout), ("id+ood pooled", pooled_scores, pooled)]:
            gate = (d["n_visible"] >= N_VISIBLE_MIN) & np.isfinite(d["mu_err_mean"]) & np.isfinite(s)
            coverage, risk, _ = risk_coverage(s[gate], d["mu_err_mean"][gate])
            curves[label] = (coverage, risk)
        plot_risk_coverage(curves, name, os.path.join(out_dir, f"risk_coverage_{name}.png"))

    detection = pd.DataFrame(detection_rows)
    risk = pd.DataFrame(risk_rows)
    detection.to_csv(os.path.join(out_dir, "extended_detection.csv"), index=False)
    risk.to_csv(os.path.join(out_dir, "extended_risk.csv"), index=False)
    print(detection.to_string(index=False))
    print()
    print(risk.to_string(index=False))
    print(f"\nresults -> {out_dir}/extended_{{detection,risk}}.csv")


if __name__ == "__main__":
    main()
