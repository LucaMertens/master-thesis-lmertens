"""Score-histogram and ROC plots for one detector/tier pair."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve


def plot_score_hist(id_scores, ood_scores, title, path):
    id_scores = id_scores[np.isfinite(id_scores)]
    ood_scores = ood_scores[np.isfinite(ood_scores)]
    lo = min(id_scores.min(), ood_scores.min())
    hi = max(id_scores.max(), ood_scores.max())
    bins = np.linspace(lo, hi, 41)
    plt.figure()
    plt.hist(id_scores, bins=bins, alpha=0.6, density=True, label=f"ID (n={len(id_scores)})")
    plt.hist(ood_scores, bins=bins, alpha=0.6, density=True, label=f"OOD (n={len(ood_scores)})")
    plt.xlabel("OOD score")
    plt.ylabel("density")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close("all")


def plot_roc(id_scores, ood_scores, title, path):
    id_scores = id_scores[np.isfinite(id_scores)]
    ood_scores = ood_scores[np.isfinite(ood_scores)]
    y = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    s = np.concatenate([id_scores, ood_scores])
    fpr, tpr, _ = roc_curve(y, s)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], "--", color="grey", lw=0.8)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close("all")
