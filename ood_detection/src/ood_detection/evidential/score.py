"""Deep Evidential Regression OOD score: epistemic uncertainty Var[gamma] =
beta / (v * (alpha - 1)) from the NIG head (training_network...trainer.py
TrainingModule._evidential_loss), pooled over each window's visible+observable
path cells during feature extraction. Requires a checkpoint trained with
MODEL.EVIDENTIAL: True (configs/grandtour_evidential.yaml)."""


def evidential_score(data, rgbonly=False):
    key = "evidential_unc_rgbonly" if rgbonly else "evidential_unc"
    return data[key]


def has_evidential(data):
    import numpy as np

    return not np.all(np.isnan(data["evidential_unc"]))
