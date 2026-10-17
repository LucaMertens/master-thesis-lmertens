"""AUROC/FPR95/AUPR for OOD detection: ID is the negative class, OOD the positive, higher = more OOD."""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def fpr_at_tpr(id_scores, ood_scores, tpr=0.95):
    """False-positive rate on ID at the threshold that catches `tpr` of OOD."""
    thresh = np.quantile(ood_scores, 1 - tpr)
    return float((id_scores >= thresh).mean())


def auroc(id_scores, ood_scores):
    y = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    s = np.concatenate([id_scores, ood_scores])
    return float(roc_auc_score(y, s))


def aupr(id_scores, ood_scores):
    """Average precision; depends on the OOD share, so report the base rate next to it."""
    y = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    s = np.concatenate([id_scores, ood_scores])
    return float(average_precision_score(y, s))


def auroc_per_mission(id_scores, ood_scores, ood_mission):
    """Mean of each OOD mission's AUROC against the pooled ID set (pooled AUROC is dominated by the longest mission)."""
    finite = np.isfinite(ood_scores)
    ood_scores, ood_mission = ood_scores[finite], ood_mission[finite]
    id_scores = id_scores[np.isfinite(id_scores)]
    per = [auroc(id_scores, ood_scores[ood_mission == m]) for m in np.unique(ood_mission)]
    return float(np.mean(per)) if per else float("nan")


def evaluate(id_scores, ood_scores):
    id_scores, ood_scores = np.asarray(id_scores), np.asarray(ood_scores)
    id_scores, ood_scores = id_scores[np.isfinite(id_scores)], ood_scores[np.isfinite(ood_scores)]
    if len(id_scores) == 0 or len(ood_scores) == 0:
        return {"auroc": float("nan"), "fpr95": float("nan"), "aupr": float("nan"),
                "base_rate": float("nan"), "n_id": len(id_scores), "n_ood": len(ood_scores)}
    return {
        "auroc": auroc(id_scores, ood_scores),
        "fpr95": fpr_at_tpr(id_scores, ood_scores),
        "aupr": aupr(id_scores, ood_scores),
        "base_rate": len(ood_scores) / (len(id_scores) + len(ood_scores)),
        "n_id": len(id_scores),
        "n_ood": len(ood_scores),
    }
