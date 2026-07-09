"""AUROC/FPR95 for OOD detection: id_scores are the negative class (in-distribution,
label 0), ood_scores the positive class (label 1); higher score = more OOD."""

import numpy as np
from sklearn.metrics import roc_auc_score


def fpr_at_tpr(id_scores, ood_scores, tpr=0.95):
    """FPR95: false-positive rate on ID when the threshold is set to catch `tpr`
    fraction of OOD samples."""
    thresh = np.quantile(ood_scores, 1 - tpr)
    return float((id_scores >= thresh).mean())


def auroc(id_scores, ood_scores):
    y = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    s = np.concatenate([id_scores, ood_scores])
    return float(roc_auc_score(y, s))


def evaluate(id_scores, ood_scores):
    id_scores, ood_scores = np.asarray(id_scores), np.asarray(ood_scores)
    id_scores, ood_scores = id_scores[np.isfinite(id_scores)], ood_scores[np.isfinite(ood_scores)]
    if len(id_scores) == 0 or len(ood_scores) == 0:
        return {"auroc": float("nan"), "fpr95": float("nan"), "n_id": len(id_scores), "n_ood": len(ood_scores)}
    return {
        "auroc": auroc(id_scores, ood_scores),
        "fpr95": fpr_at_tpr(id_scores, ood_scores),
        "n_id": len(id_scores),
        "n_ood": len(ood_scores),
    }
