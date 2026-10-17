"""Evidential OOD score: epistemic uncertainty beta / (v * (alpha - 1)) of the NIG head.

Needs a checkpoint trained with MODEL.EVIDENTIAL."""


def evidential_score(data, rgbonly=False):
    key = "evidential_unc_rgbonly" if rgbonly else "evidential_unc"
    return data[key]


def has_evidential(data):
    import numpy as np

    return not np.all(np.isnan(data["evidential_unc"]))
