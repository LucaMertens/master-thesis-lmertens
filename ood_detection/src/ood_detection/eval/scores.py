"""Raw OOD score distributions per tier, read against the ID spread."""

import argparse
import os

import numpy as np
import pandas as pd

from ood_detection.eval.run import OOD_TIERS, build_detectors, load_tier, score
from ood_detection.eval.metrics import auroc

QUANTILES = (0.1, 0.5, 0.9)


def id_scale(s_id, lo=0.01, hi=0.99):
    """(low, high) ID score quantiles, a colour scale shared across tiers."""
    s = s_id[np.isfinite(s_id)]
    return float(np.quantile(s, lo)), float(np.quantile(s, hi))


def tier_scores(features_dir, detectors, feat_keys, tiers, names):
    out = {}
    for tier in tiers:
        data = load_tier(features_dir, tier)
        if data is None:
            continue
        out[tier] = {name: score(name, data, detectors, feat_keys) for name in names}
    return out


def summary(scores_by_tier, reference):
    """One row per tier and detector: score quantiles and the tier median's percentile in ID."""
    rows = []
    for tier, per_detector in scores_by_tier.items():
        for name, s in per_detector.items():
            s = s[np.isfinite(s)]
            ref = reference[name][np.isfinite(reference[name])]
            q = np.quantile(s, QUANTILES)
            rows.append({
                "tier": tier, "detector": name, "n": len(s), "mean": s.mean(),
                **{f"p{int(p * 100)}": v for p, v in zip(QUANTILES, q)},
                "median_pct_of_id": float((ref < np.median(s)).mean()),
                "auroc_vs_id": auroc(ref, s) if tier != "id_heldout" else np.nan,
            })
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", required=True)
    ap.add_argument("--out", default=None, help="default: <features-dir>/eval/raw_scores.csv")
    ap.add_argument("--modes", type=int, default=3)
    ap.add_argument("--tiers", nargs="+", default=None)
    args = ap.parse_args(argv)

    detectors, feat_keys = build_detectors(load_tier(args.features_dir, "id_train"), args.modes)
    names = [*detectors, "viewpoint_disagreement", "modality_disagreement", "predicted_trav"]
    tiers = args.tiers or ["id_train", "id_heldout", *OOD_TIERS]
    scores_by_tier = tier_scores(args.features_dir, detectors, feat_keys, tiers, names)
    reference = scores_by_tier["id_heldout"]

    df = summary(scores_by_tier, reference)
    out = args.out or os.path.join(args.features_dir, "eval", "raw_scores.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    for name in names:
        print(f"\n== {name} ==  ID scale {tuple(round(v, 3) for v in id_scale(reference[name]))}")
        print(df[df.detector == name].drop(columns="detector").to_string(index=False))
    print(f"\nraw scores -> {out}")
    return df


if __name__ == "__main__":
    main()
