"""Compares detector rankings under novelty detection vs actual failure detection.

uv run python -m ood_detection.eval.novelty_vs_failure --features-dir <dir>
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from ood_detection.eval.extended import LOW_MU, N_VISIBLE_MIN
from ood_detection.eval.grouping import block_bootstrap, event_ids, time_blocks
from ood_detection.eval.metrics import auroc
from ood_detection.eval.run import OOD_TIERS, build_detectors, load_tier, score
from ood_detection.evidential.score import has_evidential
from ood_detection.temporal.ema import ema_by_mission

FAIL_ERR = 0.15  # "network is wrong": ~3x the median ID error, past the ID 95th pct


def pooled_windows(features_dir):
    """id_heldout + every OOD tier as one window set with a `novel` label."""
    sets = [("id_heldout", load_tier(features_dir, "id_heldout"))]
    sets += [(t, load_tier(features_dir, t)) for t in OOD_TIERS]
    sets = [(t, d) for t, d in sets if d is not None]
    keys = ("mission", "timestamp", "mu_err_mean", "mu_true_min", "n_visible")
    pooled = {k: np.concatenate([d[k] for _, d in sets]) for k in keys}
    pooled["novel"] = np.concatenate(
        [np.full(len(d["mission"]), t != "id_heldout") for t, d in sets]
    )
    return sets, pooled


def binary_auroc(scores, labels):
    if labels.all() or not labels.any():
        return float("nan")
    return auroc(scores[~labels], scores[labels])


def boot_ci(scores, labels, blocks, n_boot, seed=0):
    return block_bootstrap(
        lambda idx: binary_auroc(scores[idx], labels[idx]), blocks, n_boot, seed
    )


def analysis_row(scores, pooled, n_boot):
    gate = (
        (pooled["n_visible"] >= N_VISIBLE_MIN)
        & np.isfinite(pooled["mu_err_mean"])
        & np.isfinite(scores)
    )
    s, err = scores[gate], pooled["mu_err_mean"][gate]
    novel = pooled["novel"][gate]
    wrong = err >= FAIL_ERR
    hazard = pooled["mu_true_min"][gate] <= LOW_MU
    mission, timestamp = pooled["mission"][gate], pooled["timestamp"][gate]
    blocks = time_blocks(mission, timestamp)
    _, n_wrong_events = event_ids(mission, timestamp, wrong)
    _, n_hazard_events = event_ids(mission, timestamp, hazard)

    row = {}
    for name, labels in [("novelty", novel), ("wrong", wrong), ("hazard", hazard)]:
        row[f"auroc_{name}"] = binary_auroc(s, labels)
        lo, hi = boot_ci(s, labels, blocks, n_boot)
        row[f"auroc_{name}_lo"], row[f"auroc_{name}_hi"] = lo, hi

    onid = ~novel
    row["auroc_wrong_id"] = binary_auroc(s[onid], wrong[onid])
    lo, hi = boot_ci(s[onid], wrong[onid], blocks[onid], n_boot)
    row["auroc_wrong_id_lo"], row["auroc_wrong_id_hi"] = lo, hi
    row["spearman_err"] = float(spearmanr(s, err).statistic)
    row["spearman_err_id"] = float(spearmanr(s[onid], err[onid]).statistic)
    row.update(
        n=int(gate.sum()), n_wrong=int(wrong.sum()), n_wrong_events=n_wrong_events,
        n_hazard=int(hazard.sum()), n_hazard_events=n_hazard_events,
    )
    return row, (s, novel, wrong, gate)


def quadrant_table(detector, s, novel, wrong, id_q=0.95):
    """Alarm rate per {ID,OOD} x {right,wrong} quadrant, threshold from ID alone."""
    thresh = np.quantile(s[~novel], id_q)
    rows = []
    for is_novel in (False, True):
        for is_wrong in (False, True):
            m = (novel == is_novel) & (wrong == is_wrong)
            rows.append({
                "detector": detector,
                "quadrant": f"{'ood' if is_novel else 'id'}_{'wrong' if is_wrong else 'right'}",
                "n": int(m.sum()),
                "frac_windows": float(m.mean()),
                "alarm_rate": float((s[m] >= thresh).mean()) if m.any() else float("nan"),
            })
    return rows


def ranking_comparison(df):
    """Detector ranks under novelty AUROC vs wrong AUROC, plus Kendall tau."""
    out = df[["detector", "auroc_novelty", "auroc_wrong", "auroc_hazard"]].copy()
    out["rank_novelty"] = out["auroc_novelty"].rank(ascending=False).astype(int)
    out["rank_wrong"] = out["auroc_wrong"].rank(ascending=False).astype(int)
    tau = kendalltau(out["auroc_novelty"], out["auroc_wrong"]).statistic
    return out.sort_values("rank_novelty"), float(tau)


def plot_scatter(df, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
    for ax, ycol, title in [
        (axes[0], "auroc_wrong", f"failure = mu error >= {FAIL_ERR}"),
        (axes[1], "auroc_hazard", f"failure = hazard (mu_true_min <= {LOW_MU})"),
    ]:
        for _, r in df.iterrows():
            ax.errorbar(
                r["auroc_novelty"], r[ycol],
                xerr=[[r["auroc_novelty"] - r["auroc_novelty_lo"]], [r["auroc_novelty_hi"] - r["auroc_novelty"]]],
                yerr=[[r[ycol] - r[f"{ycol}_lo"]], [r[f"{ycol}_hi"] - r[ycol]]],
                fmt="o", ms=6, capsize=2,
            )
            dy = -12 if r["detector"].endswith("+ema") else 6
            ax.annotate(r["detector"], (r["auroc_novelty"], r[ycol]),
                        textcoords="offset points", xytext=(6, dy), fontsize=8)
        ax.plot([0.4, 1.0], [0.4, 1.0], "k--", lw=0.8, alpha=0.5)
        ax.axhline(0.5, color="gray", lw=0.6)
        ax.axvline(0.5, color="gray", lw=0.6)
        ax.set_xlabel("novelty AUROC (ID vs OOD tier)")
        ax.set_title(title, fontsize=10)
    axes[0].set_ylabel("failure AUROC (right vs wrong)")
    fig.suptitle("Detecting novelty is not detecting failure", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close("all")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", required=True)
    ap.add_argument("--out-dir", default=None, help="default: <features-dir>/eval")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args(argv)
    out_dir = args.out_dir or os.path.join(args.features_dir, "eval")
    os.makedirs(out_dir, exist_ok=True)

    id_train = load_tier(args.features_dir, "id_train")
    sets, pooled = pooled_windows(args.features_dir)
    detectors, feat_keys = build_detectors(id_train)
    names = ["mahalanobis_wayfaster", "mahalanobis_dino", "modality_disagreement"]
    if has_evidential(id_train):
        names.append("evidential")

    rows, quadrant_rows = [], []
    for name in names:
        raw = np.concatenate([score(name, d, detectors, feat_keys) for _, d in sets])
        ema = ema_by_mission(raw, pooled["mission"], pooled["timestamp"])
        for detector, scores in [(name, raw), (f"{name}+ema", ema)]:
            row, (s, novel, wrong, _) = analysis_row(scores, pooled, args.n_boot)
            rows.append({"detector": detector, **row})
            quadrant_rows += quadrant_table(detector, s, novel, wrong)

    df = pd.DataFrame(rows)
    quadrants = pd.DataFrame(quadrant_rows)
    df.to_csv(os.path.join(out_dir, "novelty_vs_failure.csv"), index=False)
    quadrants.to_csv(os.path.join(out_dir, "novelty_vs_failure_quadrants.csv"), index=False)
    plot_scatter(df, os.path.join(out_dir, "novelty_vs_failure.png"))

    ranking, tau = ranking_comparison(df)
    print(df.to_string(index=False))
    print()
    print(quadrants.to_string(index=False))
    print()
    print(ranking.to_string(index=False))
    print(f"\nKendall tau between novelty and failure rankings: {tau:.2f}")
    print(f"results -> {out_dir}/novelty_vs_failure{{,_quadrants}}.csv")
    return df, quadrants


if __name__ == "__main__":
    main()
