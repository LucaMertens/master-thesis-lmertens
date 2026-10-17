"""Exploration: per-window score distribution of every (detector, tier) pair, for k-means and tied-GMM modes.

    uv run python -m ood_detection.report.make_distributions --features-dir <cache> --out <dir>
"""

import argparse
import json
import os
import subprocess

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ood_detection.common.retier import assign  # noqa: E402
from ood_detection.eval.retiered import (METRICS, REPORTED,  # noqa: E402
                                         SPACES, aggregate, cell_distances,
                                         load_all, reference_free,
                                         usable_windows)
from ood_detection.eval.fingerprint import pairwise_null_aurocs  # noqa: E402
from ood_detection.eval.run import load_tier  # noqa: E402
from ood_detection.mahalanobis.detector import MahalanobisPP  # noqa: E402
from ood_detection.report.make_retiered_report import (DET_LABEL, DETECTORS,  # noqa: E402
                                                       PREAMBLE, TIER_LABEL, TIERS)
from ood_detection.report.plots import BAD, GOOD  # noqa: E402

ID_TIERS = ["id_train", "id_heldout"]
BINS = 36
# keeps one outlier from squashing the bins
CLIP = (1.0, 99.0)


def all_scores(data, tier, **kw):
    """detector -> (N,) window score, and the mode count the fit settled on."""
    out, picked = {}, None
    is_train = tier == "id_train"
    for space, key in [(s, k) for s, k in SPACES.items() if k in data]:
        block = data[key].astype(np.float32)
        if block.ndim == 2:
            block = block[:, None, :]
        usable = usable_windows(data, key)
        for metric in METRICS:
            det = MahalanobisPP(metric=metric, **kw).fit(
                block[is_train & usable].reshape(-1, block.shape[-1]))
            picked = det.n_modes_
            d = cell_distances(det, block)
            out[f"{space}_{metric}"] = np.where(usable, aggregate(d, *REPORTED), np.nan)
    out.update({k: np.asarray(v, dtype=float) for k, v in reference_free(data).items()})
    return out, picked


def id_summary(scores, tier, mission):
    """Typst table of ID score shape: median, IQR, skew, kurtosis and session floor."""
    held = tier == "id_heldout"
    in_id = np.isin(tier, ID_TIERS)
    rows = []
    for det in [d for d in DETECTORS if d in scores]:
        s = scores[det]
        v = s[held & np.isfinite(s)]
        z = (v - v.mean()) / v.std()
        ok = in_id & np.isfinite(s)
        rows.append((DET_LABEL[det], np.median(v),
                     np.subtract(*np.percentile(v, [75, 25])),
                     (z ** 3).mean(), (z ** 4).mean() - 3,
                     pairwise_null_aurocs(s[ok], mission[ok]).mean()))
    head = "[*median*], [*IQR*], [*skew*], [*excess kurt.*], [*session floor*]"
    body = "\n".join(f'  [{n}], [{m:.3g}], [{i:.3g}], [{sk:+.2f}], [{ku:+.2f}], [{fl:.3f}],'
                      for n, m, i, sk, ku, fl in rows)
    return (f'#table(\n  columns: (auto, ..(5 * (1fr,))), align: (left, ..(5 * (right,))),\n'
            f'  stroke: none, inset: (x: 5pt, y: 4pt),\n  table.hline(),\n  [], {head},\n'
            f'  table.hline(stroke: 0.5pt),\n{body}\n  table.hline(),\n)')


def normalize(s):
    """To [0, 1] on the 1st-99th percentile of the scored windows, clipped."""
    f = s[np.isfinite(s)]
    lo, hi = np.percentile(f, CLIP)
    return np.clip((s - lo) / (hi - lo), 0, 1) if hi > lo else np.zeros_like(s)


def figure(scores, tier, out, norm=normalize, ref="id_heldout"):
    dets = [d for d in DETECTORS if d in scores]
    cols = ID_TIERS + [t for t in TIERS if (tier == t).any()]
    fig, axes = plt.subplots(len(dets), len(cols), figsize=(0.92 * len(cols), 0.44 * len(dets)),
                             sharex=True, squeeze=False)
    edges = np.linspace(0, 1, BINS + 1)
    for i, det in enumerate(dets):
        u = norm(scores[det])
        base = u[(tier == ref) & np.isfinite(u)]
        for j, t in enumerate(cols):
            ax = axes[i][j]
            v = u[(tier == t) & np.isfinite(u)]
            # ID reference green behind, tier red on top (ID columns stay green)
            colour = GOOD if t in ID_TIERS else BAD
            ax.hist(base, bins=edges, density=True, color=GOOD, alpha=0.30, lw=0)
            if len(v):
                ax.hist(v, bins=edges, density=True, histtype="step", color=colour, lw=0.9)
                ax.axvline(np.median(v), color=colour, lw=0.7, ls=":")
            ax.set_xlim(0, 1)
            ax.set_yticks([])
            # sparse labels, adjacent panels would touch
            ax.set_xticks([0, 0.5, 1], ["0", "", "1"] if i == len(dets) - 1 else ["", "", ""])
            ax.tick_params(labelsize=5, length=2)
            for sp in ("top", "right", "left"):
                ax.spines[sp].set_visible(False)
            if i == 0:
                ax.set_title(TIER_LABEL[t].replace("-", "\n"), fontsize=5.5, pad=3)
            if j == 0:
                ax.set_ylabel(DET_LABEL[det], rotation=0, ha="right", va="center",
                              fontsize=5.5, labelpad=4)
    fig.subplots_adjust(hspace=0.4, wspace=0.16)
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)


PAGE = """
#align(center)[#text(size: 12pt)[*Per-window OOD score distributions --- {title}*]]
#figure(image("{file}", width: 100%), caption: [
  One panel per detector and tier. The green fill is {refname}, the same in every panel of
  a row; the red outline is the tier and the dotted line its median. Scores are normalized
  per detector to the 1st-99th percentile over all scored windows, so panels compare within
  a row and not between rows. Density, so panel height does not track the number of windows.
])
"""
HELD, TRAIN = "held-out in-distribution", "in-distribution train, the set the modes are fit on"
# (file, reference, page title, reference name)
VARIANTS = [
    ("kmeans16", "id_heldout", "k-means, k = 16, held-out reference", HELD),
    ("kmeans16", "id_train", "k-means, k = 16, train reference", TRAIN),
    ("gmm", "id_heldout", "tied GMM, components by BIC, held-out reference", HELD),
    ("gmm", "id_train", "tied GMM, components by BIC, train reference", TRAIN),
]
METHODS = {"kmeans16": dict(modes="kmeans", n_modes=16), "gmm": dict(modes="gmm")}
# BIC-picked mode count, persisted so a single-method rerun keeps the other title
PICKED_FILE = "bic_picked.json"


SUMMARY = """
#pagebreak()
#align(center)[#text(size: 12pt)[*In-distribution score shape*]]
#v(0.6em)
{table}
#v(0.6em)
Held-out in-distribution only, before any tier is read. Skew and excess kurtosis are zero
for a normal distribution; a heavy right tail makes the 95th percentile FPR95 reads off
unstable. The session floor is the mean folded AUROC between two in-distribution
recordings, so a high value means the reference is one distribution per recording rather
than one distribution. Scores are from {method}.
"""


def doc(picked, max_fit, summary=""):
    note = {"gmm": lambda: f" (BIC picked {picked['gmm']})" if "gmm" in picked else "",
            "kmeans16": lambda: f" (fit on {max_fit:,} cells)"}
    body = "#pagebreak()\n".join(
        PAGE.format(file=f"distributions_{m}_{r}.pdf", title=t + note[m](), refname=rn)
        for m, r, t, rn in VARIANTS)
    return PREAMBLE.replace('paper: "a4", margin: 1.9cm',
                            'paper: "a4", flipped: true, margin: 1.1cm') + body + summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-fit", type=int, default=200_000,
                    help="cells the modes and covariance are fit on")
    ap.add_argument("--methods", nargs="+", default=list(METHODS))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    data = load_all(args.features_dir)
    train = set(np.unique(load_tier(args.features_dir, "id_train")["mission"]))
    tier = assign(data["mission"], data["timestamp"], train)

    summary = ""
    pf = os.path.join(args.out, PICKED_FILE)
    picked = json.load(open(pf)) if os.path.exists(pf) else {}
    for method in args.methods:
        scores, n = all_scores(data, tier, max_fit=args.max_fit, **METHODS[method])
        if method == "gmm":
            picked[method] = n
            print(f"BIC picked {n} components")
            json.dump(picked, open(pf, "w"))
        if method == args.methods[0]:
            summary = SUMMARY.format(table=id_summary(scores, tier, data["mission"]),
                                     method=method)
        for m, ref, *_ in VARIANTS:
            if m == method:
                figure(scores, tier, os.path.join(args.out, f"distributions_{m}_{ref}.pdf"),
                       ref=ref)
    path = os.path.join(args.out, "ood_distributions.typ")
    with open(path, "w") as f:
        f.write(doc(picked, args.max_fit, summary))
    subprocess.run(["typst", "compile", path], check=True)
    print(f"doc -> {path.replace('.typ', '.pdf')}")


if __name__ == "__main__":
    main()
