"""RQ5: |mu_pred(RGB+D) - mu_pred(RGB-only)| per window, both computed in features.extract."""

import numpy as np


def disagreement_score(data, rgbonly_key="mu_pred_rgbonly"):
    return np.abs(data["mu_pred"] - data[rgbonly_key])
