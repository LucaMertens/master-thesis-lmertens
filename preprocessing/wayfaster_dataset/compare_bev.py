#!/usr/bin/env python
"""Before/after BEV mosaic comparison for the pose-graph correction, cropped at the most re-observed cells."""

import argparse
import pathlib

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shared_utils.paths import DATALOCAL_ROOT


def load(path):
    z = np.load(path)
    rgb, cnt, extent = z["rgb"], z["cnt"], list(z["extent"])
    img = np.zeros(rgb.shape, np.uint8)
    f = cnt > 0
    img[f] = np.clip(rgb[f] / cnt[f, None], 0, 255).astype(np.uint8)
    return img, cnt, extent


def common_window(a, b):
    """Crop two mosaics to their shared world region."""
    res = (a[2][1] - a[2][0]) / a[0].shape[1]
    x0, x1 = max(a[2][0], b[2][0]), min(a[2][1], b[2][1])
    y0, y1 = max(a[2][2], b[2][2]), min(a[2][3], b[2][3])
    nx, ny = int(round((x1 - x0) / res)), int(round((y1 - y0) / res))

    def cut(img, cnt, ext):
        ix, iy = int(round((x0 - ext[0]) / res)), int(round((y0 - ext[2]) / res))
        return img[iy : iy + ny, ix : ix + nx], cnt[iy : iy + ny, ix : ix + nx]

    return cut(*a), cut(*b), [x0, x1, y0, y1]


def sharpness(img, mask):
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx, gy = np.gradient(g)
    return np.sqrt(gx**2 + gy**2)[mask].mean() if mask.any() else 0.0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    root = DATALOCAL_ROOT / "wayfaster_split_videos"
    p.add_argument("--before", type=pathlib.Path, default=root / "bev.npz")
    p.add_argument("--after", type=pathlib.Path, default=root / "bev_opt.npz")
    p.add_argument("--half", type=int, default=70, help="Crop half-size in cells.")
    p.add_argument("--n-crops", type=int, default=3)
    p.add_argument("--out", type=pathlib.Path, default=root / "bev_compare.png")
    args = p.parse_args()

    A, cntA, extA = load(args.before)
    B, cntB, extB = load(args.after)
    (A, cntA), (B, cntB), ext = common_window((A, cntA, extA), (B, cntB, extB))

    print(f"{'':8}{'coverage':>10}{'sharpness':>11}")
    for tag, img, cnt in [("before", A, cntA), ("after", B, cntB)]:
        m = cnt > 0
        print(f"{tag:8}{100*m.mean():9.1f}%{sharpness(img, m):11.2f}")

    # crops on the most re-observed, well-separated locations
    score = cv2.GaussianBlur(np.minimum(cntA, cntB).astype(np.float32), (0, 0), 12)
    h, w = score.shape
    centers = []
    s = score.copy()
    for _ in range(args.n_crops):
        y, x = np.unravel_index(np.argmax(s), s.shape)
        centers.append((y, x))
        y0, y1 = max(0, y - 3 * args.half), min(h, y + 3 * args.half)
        x0, x1 = max(0, x - 3 * args.half), min(w, x + 3 * args.half)
        s[y0:y1, x0:x1] = 0

    n = len(centers)
    fig, ax = plt.subplots(2, n + 1, figsize=(5 * (n + 1), 10))
    for col, (img, tag) in enumerate([(A, "before"), (B, "after")]):
        ax[col, 0].imshow(img, origin="lower", extent=ext, interpolation="nearest")
        ax[col, 0].set_title(f"{tag}: full mosaic")
        ax[col, 0].set_aspect("equal")
        for k, (y, x) in enumerate(centers):
            y0, y1 = max(0, y - args.half), min(h, y + args.half)
            x0, x1 = max(0, x - args.half), min(w, x + args.half)
            crop = img[y0:y1, x0:x1]
            a = ax[col, k + 1]
            a.imshow(crop, origin="lower", interpolation="nearest")
            sh = sharpness(crop, np.ones(crop.shape[:2], bool))
            a.set_title(f"{tag}: crop {k+1}  sharpness {sh:.1f}")
            a.set_xticks([])
            a.set_yticks([])
    plt.tight_layout()
    plt.savefig(args.out, dpi=140)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
