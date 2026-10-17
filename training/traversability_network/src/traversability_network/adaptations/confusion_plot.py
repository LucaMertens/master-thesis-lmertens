"""Row-normalized mu-bin confusion matrix figures for wandb and the thesis."""

import os

import numpy as np


def _fmt(n):
    n = int(n)
    if n >= 1_000_000:
        return f"{n / 1e6:.1f}M"
    if n >= 1000:
        return f"{n / 1e3:.0f}k"
    return str(n)


def bin_recalls(counts):
    """Macro-averaged exact-bin and +-1-bin recall over the true bins that occur."""
    counts = np.asarray(counts, dtype=float)
    nb = counts.shape[0]
    row = counts.sum(1, keepdims=True)
    norm = counts / np.maximum(row, 1)
    present = [i for i in range(nb) if row[i, 0] > 0]
    if not present:
        return 0.0, 0.0
    exact = float(np.mean([norm[i, i] for i in present]))
    within1 = float(np.mean([
        norm[i, max(0, i - 1) : min(nb - 1, i + 1) + 1].sum() for i in present
    ]))
    return exact, within1


def training_title(configs, run_name, epoch):
    """Architecture/train-set/valid-set labels, same layout as the thesis figures."""
    if getattr(configs.DATASET, "NAME", "grandtour") == "wayfaster":
        dataset = "WayFASTER original dataset"
        train = [os.path.basename(os.path.dirname(d)) for d in configs.DATASET.TRAIN_DIRS]
        valid = [os.path.basename(os.path.dirname(d)) for d in configs.DATASET.VALID_DIRS]
    else:
        dataset = "GrandTour"
        train = list(configs.DATASET.TRAIN_MISSIONS)
        valid = list(configs.DATASET.VALID_MISSIONS)
    architecture = f"{run_name} ({configs.TAG}) on {dataset}, epoch {epoch}"
    train_set = ", ".join(train)
    valid_set = f"{', '.join(valid)} (visible cells)"
    return architecture, train_set, valid_set


def _fits(fig, text, fontsize, margin=0.94):
    """Whether a one-line text at fontsize fits within the figure width."""
    t = fig.text(0.5, 0.5, text, fontsize=fontsize)
    width = t.get_window_extent(renderer=fig.canvas.get_renderer()).width
    t.remove()
    return width <= fig.bbox.width * margin


def draw_confusion(ax, counts, means, fontsize=7, ms=7):
    """One confusion panel on ax. Returns the row totals."""
    counts = np.asarray(counts, dtype=float)
    nb = counts.shape[0]
    bw = 1.0 / nb
    row = counts.sum(1, keepdims=True)
    norm = counts / np.maximum(row, 1)

    ax.imshow(norm, cmap="viridis", vmin=0, vmax=1)
    for i in range(nb):
        for j in range(nb):
            if norm[i, j] >= 0.005:
                ax.text(j, i, f"{norm[i, j] * 100:.0f}", ha="center", va="center",
                        color="white" if norm[i, j] < 0.6 else "black", fontsize=fontsize)
    ax.plot([-0.5, nb - 0.5], [-0.5, nb - 0.5], color="red", lw=0.8, alpha=0.5,
            label="Perfect calibration")
    ys = [i for i in range(nb) if row[i, 0] > 0 and means[i] is not None]
    ax.plot([means[i] / bw - 0.5 for i in ys], ys, "-o", color="orange", mec="black",
            mew=0.6, ms=ms, lw=1.5, label="Mean prediction per true bin")
    return row.squeeze(1)


# half the thesis text width
PANEL_W = 3.15
PANEL_H = 3.7


def panel_title(counts, architecture=None, train_set=None, valid_set=None, heading=None):
    """Given fields (or a short heading) above the recall line."""
    diag, w1 = bin_recalls(counts)
    lines = [heading] if heading else []
    for label, field in (("Architecture", architecture), ("Train set", train_set),
                         ("Validation set", valid_set)):
        if field:
            lines.append(f"{label}: {field}")
    return "\n".join(lines + [f"Exact-bin recall = {diag:.3f},  $\\pm$1 bin = {w1:.3f}"])


def render_panels(panels, fontsize=8, tick=6, suptitle=None, figsize=None, cell=5):
    """A row of confusion panels sharing font size and heading block.

    panels: dicts of counts, means and either architecture/train_set/valid_set or heading."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nb = np.asarray(panels[0]["counts"]).shape[0]
    labels = [f"{i / nb:.1f}" for i in range(nb)]
    fig, axes = plt.subplots(1, len(panels), squeeze=False, layout="constrained",
                             figsize=figsize or (PANEL_W * len(panels), PANEL_H))
    rows = []
    for ax, p in zip(axes[0], panels):
        row = draw_confusion(ax, p["counts"], p["means"], fontsize=cell, ms=3.5)
        rows.append(row)
        ax.set_title(panel_title(p["counts"], p.get("architecture"), p.get("train_set"),
                                 p.get("valid_set"), p.get("heading")),
                     fontsize=fontsize, pad=3)
        ax.set_xticks(range(nb), labels, fontsize=tick)
        # panels on the same population repeat the row totals; label only the leftmost
        shared = np.array_equal(row, rows[0]) and ax is not axes[0][0]
        ax.set_yticks(range(nb), [""] * nb if shared else
                      [f"{labels[i]} (n={_fmt(row[i])})" for i in range(nb)],
                      fontsize=tick)
        ax.set_xlabel("Predicted $\\mu$ bin", fontsize=fontsize)

    axes[0][0].set_ylabel("True $\\mu$ bin (row total)", fontsize=fontsize)
    if suptitle:
        fig.suptitle(suptitle, fontsize=fontsize)
    fig.legend(*axes[0][0].get_legend_handles_labels(), loc="outside lower center",
               ncol=2, fontsize=7, framealpha=0.9)
    return fig


def render_confusion(counts, means, architecture=None, train_set=None, valid_set=None):
    """counts: (NB,NB) true-bin x pred-bin. means: per-true-bin mean pred mu
    (None for empty bins). Title lines are omitted when their field is falsy."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nb = np.asarray(counts).shape[0]
    bw = 1.0 / nb
    labels = [f"{i * bw:.1f}" for i in range(nb)]
    diag, w1 = bin_recalls(counts)

    fig, ax = plt.subplots(figsize=(7.2, 6.4), layout="constrained")
    row = draw_confusion(ax, counts, means)
    ax.set_xticks(range(nb))
    ax.set_yticks(range(nb))
    ax.set_xticklabels(labels)
    ax.set_yticklabels([f"{labels[i]}  (n={_fmt(row[i])})" for i in range(nb)])
    ax.set_xlabel("Predicted $\\mu$ bin (linear traversability)")
    ax.set_ylabel("True $\\mu$ bin (row total)")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.09),
        ncol=2,
        fontsize=8,
        framealpha=0.9,
    )
    stats = f"Exact-bin recall = {diag:.3f},  $\\pm$1 bin = {w1:.3f}"
    lines = [f"Architecture: {architecture}"] if architecture else []
    if train_set and valid_set:
        combined = f"Train set: {train_set}   |   Validation set: {valid_set}"
        lines.append(
            combined
            if _fits(fig, combined, fontsize=9)
            else f"Train set: {train_set}\nValidation set: {valid_set}"
        )
    elif train_set:
        lines.append(f"Train set: {train_set}")
    elif valid_set:
        lines.append(f"Validation set: {valid_set}")
    lines.append(stats)
    fig.suptitle("\n".join(lines), fontsize=9)
    return fig
