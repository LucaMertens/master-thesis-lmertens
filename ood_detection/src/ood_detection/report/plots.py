"""The two figures of the OOD report: in-distribution typicality and detection."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

TEXTWIDTH = 6.30

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
})

BAD, GOOD, FLOOR = "#c44e52", "#55a868", "#4c72b0"
BEV_C, DINO_C, FREE_C = "#c76b28", "#3b6ea5", "#5c5c5c"

# One distance per family plus the reference-free detectors; cosine and predicted_trav stay in the tables
SERIES = [
    ("bev_mahalanobis", "BEV Mahalanobis++", BEV_C, "-", "o"),
    ("bev_euclidean", "BEV k-means", BEV_C, "--", "^"),
    ("dino_mahalanobis", "DINOv3 Mahalanobis++", DINO_C, "-", "o"),
    ("dino_euclidean", "DINOv3 k-means", DINO_C, "--", "^"),
    ("viewpoint_disagreement", "viewpoint disagreement", FREE_C, "-", "s"),
    ("viewpoint_variance", "viewpoint variance", FREE_C, "-.", "P"),
    ("modality_disagreement", "modality disagreement", FREE_C, "--", "s"),
    ("depth_entropy", "depth entropy", FREE_C, "-", "D"),
]


def family_handles():
    return [plt.Line2D([], [], ls="", marker="o", ms=4.5, mfc="none", mec=BEV_C,
                       mew=1.1, label="BEV distance"),
            plt.Line2D([], [], ls="", marker="o", ms=4.5, color=DINO_C,
                       label="DINOv3 distance"),
            plt.Line2D([], [], ls="", marker="o", ms=4.5, color=FREE_C,
                       label="reference-free")]


def typicality_figure(typ, out):
    """One row per ID mission, one dot per detector. Right of 0.5 is the bad half."""
    wide = typ.pivot_table(index=["split", "mission"], columns="detector", values="auroc_vs_id")
    held = wide.loc["heldout"]
    held = held.reindex(held.mean(axis=1).sort_values(ascending=False).index)
    train = wide.loc["train"]
    train = train.reindex(train.mean(axis=1).sort_values(ascending=False).index)

    rows = ([("head", "train (inside the fit set)")] + [("m", m) for m in train.index]
            + [("head", "held out")] + [("m", m) for m in held.index])
    y = {i: len(rows) - 1 - i for i in range(len(rows))}

    fig, ax = plt.subplots(figsize=(TEXTWIDTH, 0.235 * len(rows) + 1.15))
    ax.axvspan(0.5, 1.0, color=BAD, alpha=0.07, lw=0)
    ax.axvline(0.5, color=BAD, lw=1.0, zorder=1)

    for i, (kind, label) in enumerate(rows):
        if kind == "head":
            ax.text(0.0, y[i], label, ha="left", va="center", fontsize=7.5,
                    style="italic", color="0.35")
            continue
        row = (held if label in held.index else train).loc[label]
        for det in row.index:
            c = (DINO_C if det.startswith("dino") else
                 BEV_C if det.startswith("bev") else FREE_C)
            ax.plot(row[det], y[i], "o", ms=4.5, mfc="none" if det.startswith("bev") else c,
                    mec=c, mew=1.1, zorder=3)
        ax.text(-0.012, y[i], label, ha="right", va="center", fontsize=7.5)

    top = len(rows) - 0.35
    ax.text(0.505, top, "reads as out-of-distribution", color=BAD, fontsize=7.5, va="center")
    ax.text(0.495, top, "typical for in-distribution", color="0.35", fontsize=7.5,
            va="center", ha="right")

    ax.set_yticks([])
    ax.set_ylim(-0.7, len(rows) + 0.1)
    ax.set_xlim(0, 1)
    ax.set_xlabel("AUROC of one ID mission against every other ID window")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)

    fig.legend(handles=family_handles(), loc="lower center", frameon=False, ncol=3,
               bbox_to_anchor=(0.55, -0.045))
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)


def detection_figure(res, tiers, tier_label, out):
    """Absolute AUROC, and the same values as a lift over each detector's session floor."""
    wide = res.pivot_table(index="detector", columns="tier", values="auroc")
    nulls = res.groupby("detector")["null_mean"].max()
    cols = [t for t in tiers if t in wide.columns]
    x = np.arange(len(cols))

    fig, axes = plt.subplots(2, 1, figsize=(TEXTWIDTH, 5.2), sharex=True,
                             gridspec_kw={"hspace": 0.16})
    for key, label, colour, ls, marker in SERIES:
        if key not in wide.index:
            continue
        row = wide.loc[key]
        axes[0].plot(x, [row[c] for c in cols], ls=ls, marker=marker, ms=4, lw=1.1,
                     color=colour, mfc="none" if ls == "--" else colour, label=label)
        base = nulls[key]
        axes[1].plot(x, [row[c] - base for c in cols], ls=ls, marker=marker, ms=4, lw=1.1,
                     color=colour, mfc="none" if ls == "--" else colour)

    axes[0].axhline(0.5, color="0.6", lw=0.8, ls=":")
    axes[0].text(len(cols) - 0.45, 0.5, "chance", fontsize=7, color="0.45", va="bottom",
                 ha="right")
    axes[0].set_ylabel("AUROC")
    shown = [k for k, *_ in SERIES if k in wide.index]
    axes[0].set_ylim(min(0.2, wide.loc[shown, cols].min().min() - 0.05), 1.06)
    axes[0].annotate("higher is better", xy=(0.008, 0.965), xycoords="axes fraction",
                     fontsize=7.5, color=GOOD)

    axes[1].axhline(0, color=FLOOR, lw=1.2)
    axes[1].text(-0.42, 0.012, "session floor: two in-distribution recordings, no shift",
                 fontsize=7.5, color=FLOOR, va="bottom",
                 bbox=dict(facecolor="white", edgecolor="none", pad=1.0))
    axes[1].set_ylabel("AUROC $-$ session floor")
    axes[1].annotate("above 0 = detects the shift, not the recording",
                     xy=(0.008, 0.955), xycoords="axes fraction", fontsize=7.5, color=GOOD)

    for ax in axes:
        ax.set_xticks(x, [tier_label[c] for c in cols])
        ax.grid(axis="y", lw=0.4, color="0.9")
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    # clear the three-line tick labels
    fig.legend(handles, labels, loc="lower center", frameon=False, ncol=3,
               handlelength=2.4, bbox_to_anchor=(0.55, -0.135))
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)


def aggregator_figure(sweep, out):
    """Mean AUROC over tiers per cell-distance quantile, with the mean alongside. BEV only (DINOv3 is pooled)."""
    g = sweep[sweep["space"] == "bev"].groupby(
        ["metric", "kind", "q"])["auroc"].mean().reset_index()
    name = {"mahalanobis": "Mahalanobis++", "euclidean": "k-means", "cosine": "cosine"}
    colour = {"mahalanobis": "#c44e52", "euclidean": "#c76b28", "cosine": "#8172b2"}

    fig, ax = plt.subplots(figsize=(TEXTWIDTH, 2.9))
    for metric, d in g.groupby("metric"):
        c = colour[metric]
        q = d[d["kind"] == "quantile"].sort_values("q")
        ax.plot(q["q"], q["auroc"], "-", marker="o", ms=3.2, lw=1.1, color=c,
                label=name[metric])
        ax.axhline(float(d[d["kind"] == "mean"]["auroc"].iloc[0]), color=c, lw=0.9, ls=":")
        med = q[q["q"] == 0.5]
        ax.plot(med["q"], med["auroc"], "*", ms=9, color=c, zorder=4)

    ax.axhline(0.5, color="0.6", lw=0.8, ls="--")
    ax.text(0.02, 0.5, "chance", fontsize=7, color="0.45", va="bottom")
    ax.plot([], [], "*", ms=9, color="0.35", label="median, the reported score")
    ax.plot([], [], ls=":", color="0.35", lw=0.9, label="mean of all cells")
    ax.set_xlabel("quantile of the window's cell distances")
    ax.set_ylabel("mean AUROC over tiers")
    ax.set_xlim(0, 1.02)
    ax.grid(lw=0.4, color="0.9")
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.legend(loc="lower center", frameon=False, ncol=3, bbox_to_anchor=(0.55, -0.20))
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)


def typicality_all_figure(typ_all, tiers, tier_label, out):
    """Typicality of every mission, grouped by tier (a mission split across tiers gets one row per part)."""
    wide = typ_all.pivot_table(index=["tier", "mission"], columns="detector",
                               values="auroc_vs_id")
    rows = []
    for t in tiers:
        if t not in wide.index.get_level_values(0):
            continue
        block = wide.loc[t]
        block = block.reindex(block.mean(axis=1).sort_values(ascending=False).index)
        rows.append(("head", tier_label.get(t, t), None))
        rows += [("m", m, block.loc[m]) for m in block.index]
    y = {i: len(rows) - 1 - i for i in range(len(rows))}

    fig, ax = plt.subplots(figsize=(TEXTWIDTH, 0.205 * len(rows) + 1.15))
    ax.axvspan(0.5, 1.0, color=BAD, alpha=0.07, lw=0)
    ax.axvline(0.5, color=BAD, lw=1.0, zorder=1)

    for i, (kind, label, row) in enumerate(rows):
        if kind == "head":
            ax.text(0.0, y[i], label, ha="left", va="center", fontsize=7.5,
                    style="italic", color="0.35")
            continue
        for det in row.index:
            c = (DINO_C if det.startswith("dino") else
                 BEV_C if det.startswith("bev") else FREE_C)
            ax.plot(row[det], y[i], "o", ms=4.0, mfc="none" if det.startswith("bev") else c,
                    mec=c, mew=1.0, zorder=3)
        ax.text(-0.012, y[i], label, ha="right", va="center", fontsize=7.0)

    top = len(rows) - 0.35
    ax.text(0.505, top, "separable from in-distribution", color=BAD, fontsize=7.5, va="center")
    ax.text(0.495, top, "reads as in-distribution", color="0.35", fontsize=7.5,
            va="center", ha="right")

    ax.set_yticks([])
    ax.set_ylim(-0.7, len(rows) + 0.1)
    ax.set_xlim(0, 1)
    ax.set_xlabel("AUROC of one mission or mission part against every in-distribution window")
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)

    fig.legend(handles=family_handles(), loc="lower center", frameon=False, ncol=3,
               bbox_to_anchor=(0.55, -0.03))
    fig.savefig(out, bbox_inches="tight")


# Hand-tuned label offsets in points (dx, dy, ha) so clustered labels do not overlap; default is up-right
LABEL_NUDGE = {
    "dino_euclidean": (-10, 2, "right"), "dino_mahalanobis": (-10, 2, "right"),
    "viewpoint_disagreement": (9, 2, "left"), "viewpoint_variance": (-2, -14, "left"),
    "modality_disagreement": (9, -2, "left"), "depth_entropy": (9, 1, "left"),
}
LABEL_NUDGE_RAW = {
    "dino_euclidean": (-10, 5, "right"), "dino_mahalanobis": (-10, -10, "right"),
    "depth_entropy": (-10, 2, "right"), "viewpoint_variance": (0, 10, "center"),
    "viewpoint_disagreement": (0, -13, "center"), "modality_disagreement": (9, -2, "left"),
}


def shift_class_figure(res, nulls, classes, class_label, det_label, out, note="",
                       baseline=True):
    """Performance on terrain-change tiers vs view-only tiers (above the diagonal = better on hazards).

    `baseline` subtracts each detector's session floor; otherwise the reference is chance.
    """
    hazard, benign = classes["silent_hazard"], classes["benign_novelty"]
    wide = res.pivot_table(index="detector", columns="tier", values="auroc")
    lift = wide.sub(pd.Series(nulls), axis=0) if baseline else wide
    ref = 0.0 if baseline else 0.5
    nudge = LABEL_NUDGE if baseline else LABEL_NUDGE_RAW
    pts = {k: (lift.loc[k, benign].mean(), lift.loc[k, hazard].mean())
           for k, *_ in SERIES if k in lift.index}

    fig, ax = plt.subplots(figsize=(TEXTWIDTH * 0.78, TEXTWIDTH * 0.62))
    lo = min(min(v) for v in pts.values()) - 0.06
    hi = max(max(v) for v in pts.values()) + 0.10
    # useful only above the reference and above the diagonal
    xs = np.linspace(lo, hi, 200)
    ax.fill_between(xs, np.maximum(xs, ref), hi, where=np.maximum(xs, ref) < hi,
                    color=GOOD, alpha=0.08, lw=0)
    ax.plot([lo, hi], [lo, hi], color="0.6", lw=0.8, ls="--", zorder=1)
    ax.axhline(ref, color=FLOOR, lw=1.0, zorder=1)
    ax.axvline(ref, color=FLOOR, lw=1.0, zorder=1)

    for key, label, colour, ls, marker in SERIES:
        if key not in pts:
            continue
        x, y = pts[key]
        ax.plot(x, y, marker=marker, ms=7, color=colour,
                mfc="none" if ls == "--" else colour, mew=1.4, ls="", zorder=3)
        dx, dy, ha = nudge.get(key, (9, 3, "left"))
        ax.annotate(det_label[key], (x, y), textcoords="offset points", xytext=(dx, dy),
                    fontsize=7.5, color=colour, ha=ha)

    ax.text(lo + 0.01, hi - 0.01, "better at hidden hazards", color=GOOD, fontsize=7.5,
            va="top", ha="left")
    ax.text(hi - 0.01, lo + 0.01, "better at new scenery", color="0.45", fontsize=7.5,
            va="bottom", ha="right")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(f"performance on {class_label['benign_novelty']} tiers")
    ax.set_ylabel(f"performance on {class_label['silent_hazard']} tiers")
    # figure is used standalone, so it defines its own classes and baseline
    if note:
        fig.text(0.5, -0.055, note, ha="center", va="top", fontsize=7.5, color="0.35")
    ax.grid(lw=0.4, color="0.92")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)
