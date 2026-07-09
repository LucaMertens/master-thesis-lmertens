"""Optional RQ5: modality disagreement as a cheap pseudo-ensemble OOD score.
Training already applies 30% point-cloud dropout (AUGMENTATIONS.PCLOUD_DROPOUT),
so the RGB-only path is a genuine second predictor with zero extra parameters or
training. |mu_pred(RGB+D) - mu_pred(RGB-only)| per window; both columns are
computed once during feature extraction (ood_detection.features.extract)."""

import numpy as np


def disagreement_score(data, rgbonly_key="mu_pred_rgbonly"):
    return np.abs(data["mu_pred"] - data[rgbonly_key])
