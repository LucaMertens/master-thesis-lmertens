"""ForEnt incident windows: hand-labelled entrapment spans plus a short run-up."""

import numpy as np

from paths import FORENT_EXTRACTED_ROOT

FORENT_TIER = "forent_incident"
FORENT_LEAD_S = 3.0


def forent_incident(mission, timestamp, lead_s=FORENT_LEAD_S):
    """(N,) True inside a labelled entrapment or in the `lead_s` seconds before one."""
    out = np.zeros(len(mission), bool)
    for m in np.unique(mission):
        path = FORENT_EXTRACTED_ROOT / str(m) / "entrapment.npy"
        if not path.exists():
            continue
        spans = np.load(path)
        sel = mission == m
        t = timestamp[sel][:, None]
        out[sel] = ((t >= spans[None, :, 0] - lead_s) & (t <= spans[None, :, 1])).any(1)
    return out
