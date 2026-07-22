"""OOD tier AUROC corrected for recording-session fingerprint bias.

uv run python -m ood_detection.eval.fingerprint --features-dir <dir>
"""

import argparse
import os

import numpy as np
import pandas as pd

from ood_detection.eval.metrics import auroc
from ood_detection.eval.run import OOD_TIERS, build_detectors, load_tier, score
from ood_detection.evidential.score import has_evidential


def finite(scores, mission):
    m = np.isfinite(scores)
    return scores[m], mission[m]


def pairwise_null_aurocs(scores, mission):
    """Folded pairwise AUROC between same-domain missions: the score separation
    that session identity alone produces."""
    ms = np.unique(mission)
    pairs = [
        auroc(scores[mission == a], scores[mission == b])
        for i, a in enumerate(ms) for b in ms[i + 1:]
    ]
    return np.array([max(a, 1 - a) for a in pairs])


def per_mission_aurocs(id_scores, ood_scores, ood_mission):
    """Directed AUROC of each OOD mission against the pooled ID set."""
    return {m: auroc(id_scores, ood_scores[ood_mission == m]) for m in np.unique(ood_mission)}


def beats_null(mission_aurocs, null_aurocs):
    """P(an OOD mission's AUROC exceeds an ID-vs-ID session pair's AUROC).
    0.5 = the tier shift is indistinguishable from session fingerprint."""
    if len(null_aurocs) == 0 or len(mission_aurocs) == 0:
        return float("nan")
    a = np.asarray(mission_aurocs)[:, None]
    b = np.asarray(null_aurocs)[None, :]
    return float(((a > b) + 0.5 * (a == b)).mean())


def centered(scores, mission):
    """Remove each recording's level: subtract the per-mission median."""
    out = scores.astype(np.float64).copy()
    for m in np.unique(mission):
        idx = mission == m
        out[idx] -= np.median(scores[idx])
    return out


def level_auroc(id_scores, id_mission, ood_scores, ood_mission):
    """AUROC over per-mission median scores only (one number per recording)."""
    id_med = [np.median(id_scores[id_mission == m]) for m in np.unique(id_mission)]
    ood_med = [np.median(ood_scores[ood_mission == m]) for m in np.unique(ood_mission)]
    return auroc(np.array(id_med), np.array(ood_med))


def tier_row(id_scores, id_mission, ood_scores, ood_mission, null):
    mission_aurocs = per_mission_aurocs(id_scores, ood_scores, ood_mission)
    vals = np.array(list(mission_aurocs.values()))
    return {
        "auroc_pooled": auroc(id_scores, ood_scores),
        "auroc_mission_mean": float(vals.mean()),
        "auroc_mission_min": float(vals.min()),
        "null_mean": float(null.mean()),
        "null_max": float(null.max()),
        "beats_null": beats_null(vals, null),
        "excess": float(vals.mean() - null.mean()),
        "auroc_centered": auroc(centered(id_scores, id_mission), centered(ood_scores, ood_mission)),
        "auroc_level": level_auroc(id_scores, id_mission, ood_scores, ood_mission),
        "fingerprint_tier": float(pairwise_null_aurocs(ood_scores, ood_mission).mean())
        if len(np.unique(ood_mission)) > 1 else float("nan"),
        "n_missions": len(vals),
    }


def ranking_changes(df):
    """Per tier, detector order under pooled AUROC vs under beats_null."""
    rows = []
    for tier, g in df.groupby("tier"):
        raw = list(g.sort_values("auroc_pooled", ascending=False)["detector"])
        corrected = list(g.sort_values(["beats_null", "excess"], ascending=False)["detector"])
        rows.append({"tier": tier, "rank_pooled": " > ".join(raw),
                     "rank_beats_null": " > ".join(corrected),
                     "changed": raw != corrected})
    return pd.DataFrame(rows)


def plot_fingerprint(results, nulls, path):
    """Per detector: null pair AUROCs as a gray band, per-OOD-mission AUROCs as dots."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    detectors = list(nulls)
    fig, axes = plt.subplots(1, len(detectors), figsize=(4.2 * len(detectors), 4), sharey=True)
    axes = np.atleast_1d(axes)
    tiers = sorted({r["tier"] for r in results})
    for ax, det in zip(axes, detectors):
        null = nulls[det]
        ax.axhspan(null.min(), null.max(), color="gray", alpha=0.25,
                   label="ID x ID session null")
        ax.scatter(np.zeros(len(null)), null, color="gray", s=18, zorder=3)
        for x, tier in enumerate(tiers, start=1):
            vals = [v for r in results if r["detector"] == det and r["tier"] == tier
                    for v in r["mission_aurocs"].values()]
            ax.scatter(np.full(len(vals), x), vals, s=22, zorder=3)
        ax.set_xticks(range(len(tiers) + 1), ["null"] + tiers, rotation=30, fontsize=8)
        ax.axhline(0.5, color="k", lw=0.6)
        ax.set_title(det, fontsize=10)
        ax.set_ylim(0.45, 1.02)
    axes[0].set_ylabel("per-mission AUROC vs pooled ID")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Which tiers exceed the session-fingerprint floor?", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close("all")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", required=True)
    ap.add_argument("--out-dir", default=None, help="default: <features-dir>/eval")
    args = ap.parse_args(argv)
    out_dir = args.out_dir or os.path.join(args.features_dir, "eval")
    os.makedirs(out_dir, exist_ok=True)

    id_train = load_tier(args.features_dir, "id_train")
    id_heldout = load_tier(args.features_dir, "id_heldout")
    tiers = {t: d for t in OOD_TIERS if (d := load_tier(args.features_dir, t)) is not None}
    detectors, feat_keys = build_detectors(id_train)
    names = ["mahalanobis_wayfaster", "mahalanobis_dino", "modality_disagreement"]
    if has_evidential(id_train):
        names.append("evidential")

    rows, results, nulls = [], [], {}
    for name in names:
        s_id, m_id = finite(score(name, id_heldout, detectors, feat_keys), id_heldout["mission"])
        null = pairwise_null_aurocs(s_id, m_id)
        nulls[name] = null
        for tier, data in tiers.items():
            s_ood, m_ood = finite(score(name, data, detectors, feat_keys), data["mission"])
            row = tier_row(s_id, m_id, s_ood, m_ood, null)
            rows.append({"detector": name, "tier": tier, **row})
            results.append({"detector": name, "tier": tier,
                            "mission_aurocs": per_mission_aurocs(s_id, s_ood, m_ood)})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "fingerprint_corrected.csv"), index=False)
    plot_fingerprint(results, nulls, os.path.join(out_dir, "fingerprint_corrected.png"))

    per_mission = pd.DataFrame([
        {"detector": r["detector"], "tier": r["tier"], "mission": m, "auroc": a}
        for r in results for m, a in r["mission_aurocs"].items()
    ])
    per_mission.to_csv(os.path.join(out_dir, "fingerprint_per_mission.csv"), index=False)

    ranks = ranking_changes(df)
    print(df.to_string(index=False))
    print()
    print(ranks.to_string(index=False))
    print(f"\nresults -> {out_dir}/fingerprint_{{corrected,per_mission}}.csv")
    return df, ranks


if __name__ == "__main__":
    main()
