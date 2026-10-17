"""Raw .npy sidecars for the PNG caches (25 ms PNG decode vs 0.6 ms, ~3x the bytes).

Build once per cache root:

    uv run python -m traversability_network.adaptations.npy_cache <root> [<root> ...]
"""

import os
import pathlib
import sys
from multiprocessing import Pool

import cv2
import numpy as np


def imread(path, use_sidecar=True):
    """cv2.imread(path, -1), preferring the raw sidecar when one exists."""
    if use_sidecar:
        try:
            return np.load(path[:-4] + ".npy")
        except OSError:
            pass
    return cv2.imread(path, -1)


def _convert(png):
    npy = png[:-4] + ".npy"
    if os.path.exists(npy):
        return 0
    img = cv2.imread(png, -1)
    if img is None:
        return 0
    # write then rename, so readers never see partial files
    tmp = f"{npy}.{os.getpid()}.tmp"
    with open(tmp, "wb") as fh:
        np.save(fh, img)
    os.replace(tmp, npy)
    return 1


def build(roots, n_workers=None):
    """Write a sidecar next to every .png under roots. Idempotent."""
    pngs = [str(p) for root in roots for p in pathlib.Path(root).rglob("*.png")]
    with Pool(n_workers or os.cpu_count()) as pool:
        written = sum(pool.imap_unordered(_convert, pngs, chunksize=64))
    print(f"{written} sidecars written, {len(pngs) - written} already present")


if __name__ == "__main__":
    build(sys.argv[1:])
