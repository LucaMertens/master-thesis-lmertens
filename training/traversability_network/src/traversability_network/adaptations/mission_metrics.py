"""Per-mission mu metrics (calibration slope, risky-vs-rest AUROC, +-1-bin recall) scored offline from a checkpoint."""

import numpy as np
from scipy.stats import rankdata

from .confusion_plot import bin_recalls

N_BINS = 10
# Risky/probably-traversable boundary of the labeling scheme
RISKY_THRESHOLD = 0.5


def confusion(pred, true, n_bins=N_BINS):
    tb = np.clip((true * n_bins).astype(int), 0, n_bins - 1)
    pb = np.clip((pred * n_bins).astype(int), 0, n_bins - 1)
    return np.bincount(tb * n_bins + pb, minlength=n_bins**2).reshape(n_bins, n_bins).astype(float)


def calibration_slope(pred, true, n_bins=N_BINS):
    """Slope of mean predicted mu against true-mu bin centre; <2 occupied bins gives 0."""
    b = np.clip((true * n_bins).astype(int), 0, n_bins - 1)
    count = np.bincount(b, minlength=n_bins)
    total = np.bincount(b, weights=pred, minlength=n_bins)
    return slope_from_bins(count, total, n_bins)


def auroc_risky(pred, true, threshold=RISKY_THRESHOLD):
    """P(random risky cell is predicted lower than a random traversable one); ties count half."""
    risky = true < threshold
    n_risky, n_rest = int(risky.sum()), int((~risky).sum())
    if n_risky == 0 or n_rest == 0:
        return np.nan
    ranks = rankdata(-pred)  # descending prediction, so risky cells should rank first
    return float((ranks[risky].sum() - n_risky * (n_risky + 1) / 2) / (n_risky * n_rest))


def recall_within_one_bin(pred, true, n_bins=N_BINS):
    return float(bin_recalls(confusion(pred, true, n_bins))[1])


MIN_RISKY_CELLS = 200


# ---- histogram forms of the same metrics ----
# Computed from per-mission histograms (confusion, summed prediction per bin, score
# histogram) so training can score itself. Only the AUROC differs, via bin-level ties.


def slope_from_bins(count, pred_sum, n_bins=N_BINS):
    occupied = count > 0
    if occupied.sum() < 2:
        return 0.0
    x = ((np.arange(n_bins) + 0.5) / n_bins)[occupied]
    y = pred_sum[occupied] / count[occupied]
    return float(((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum())


def auroc_from_scores(risky, rest):
    """AUROC from predicted-mu histograms of the two populations, bins ascending in mu."""
    n_risky, n_rest = risky.sum(), rest.sum()
    if n_risky == 0 or n_rest == 0:
        return np.nan
    above = rest[::-1].cumsum()[::-1] - rest
    return float(((above + 0.5 * rest) * risky).sum() / (n_risky * n_rest))


def false_alarm_from_confusion(counts, threshold=RISKY_THRESHOLD, n_bins=N_BINS):
    cut = int(threshold * n_bins)
    rest = counts[cut:].sum()
    return float(counts[cut:, :cut].sum() / rest) if rest else np.nan


def score_hist(counts, pred_sum, scores, threshold=RISKY_THRESHOLD, n_bins=N_BINS):
    """One mission's metrics from its histograms; same keys as ``score``."""
    count = counts.sum(1)
    cut = int(threshold * n_bins)
    return {
        "calib_slope": slope_from_bins(count, pred_sum, n_bins),
        "auroc_risky": auroc_from_scores(scores[0], scores[1]),
        "recall_pm1": float(bin_recalls(counts)[1]),
        "false_alarm": false_alarm_from_confusion(counts, threshold, n_bins),
        "n_cells": float(count.sum()),
        "n_risky": float(count[:cut].sum()),
    }


def false_alarm(pred, true, threshold=RISKY_THRESHOLD):
    """Share of crossable cells predicted risky; defined even without risky cells."""
    rest = true >= threshold
    return float((pred[rest] < threshold).mean()) if rest.any() else np.nan


def score(pred, true, threshold=RISKY_THRESHOLD, n_bins=N_BINS):
    return {
        "calib_slope": calibration_slope(pred, true, n_bins),
        "auroc_risky": auroc_risky(pred, true, threshold),
        "recall_pm1": recall_within_one_bin(pred, true, n_bins),
        "false_alarm": false_alarm(pred, true, threshold),
        "n_cells": float(len(true)),
        "n_risky": float((true < threshold).sum()),
    }


def by_mission(pred, true, mission, threshold=RISKY_THRESHOLD, n_bins=N_BINS,
               min_risky=MIN_RISKY_CELLS):
    """(per-mission scores, average over missions that carry the class)."""
    per = {m: score(pred[mission == m], true[mission == m], threshold, n_bins)
           for m in sorted(set(mission.tolist()))}
    return per, average(per, min_risky)


def average(per, min_risky=MIN_RISKY_CELLS):
    """Equal-weight average over missions with at least min_risky risky cells.

    Low-risk missions are dropped because slope and +-1-bin recall are uninformative there."""
    scored = [v for v in per.values() if v["n_risky"] >= min_risky]
    keys = ("calib_slope", "auroc_risky", "recall_pm1")
    mean = {k: _nanmean([v[k] for v in scored]) for k in keys}
    # all missions, including the excluded ones
    mean["false_alarm"] = _nanmean([v["false_alarm"] for v in per.values()])
    mean["n_missions"] = len(per)
    mean["n_missions_scored"] = len(scored)
    mean["n_cells"] = float(sum(v["n_cells"] for v in scored))
    mean["n_risky"] = float(sum(v["n_risky"] for v in scored))
    return mean


def _nanmean(values):
    """nanmean that returns nan for empty input without warning."""
    kept = [v for v in values if not np.isnan(v)]
    return float(np.mean(kept)) if kept else np.nan
