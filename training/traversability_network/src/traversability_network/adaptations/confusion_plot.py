"""Renders the 10x10 mu-bin confusion matrix (true vs predicted bin, row-normalized recall) as a matplotlib figure for wandb."""
import numpy as np


def _fmt(n):
    n = int(n)
    if n >= 1_000_000:
        return f"{n / 1e6:.1f}M"
    if n >= 1000:
        return f"{n / 1e3:.0f}k"
    return str(n)


def render_confusion(counts, means, balanced_mae, title):
    """counts: (NB,NB) true-bin x pred-bin. means: per-true-bin mean pred mu
    (None for empty bins). Returns a matplotlib Figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = np.asarray(counts, dtype=float)
    nb = counts.shape[0]
    bw = 1.0 / nb
    labels = [f"{i * bw:.1f}" for i in range(nb)]
    row = counts.sum(1, keepdims=True)
    norm = counts / np.maximum(row, 1)
    present = row.squeeze(1) > 0

    diag = np.mean([norm[i, i] for i in range(nb) if present[i]]) if present.any() else 0.0
    w1 = []
    for i in range(nb):
        if present[i]:
            lo, hi = max(0, i - 1), min(nb - 1, i + 1)
            w1.append(norm[i, lo:hi + 1].sum())
    w1 = float(np.mean(w1)) if w1 else 0.0

    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    im = ax.imshow(norm, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(nb)); ax.set_yticks(range(nb))
    ax.set_xticklabels(labels)
    ax.set_yticklabels([f"{labels[i]}  (n={_fmt(row[i, 0])})" for i in range(nb)])
    ax.set_xlabel("predicted mu bin"); ax.set_ylabel("true mu bin  (row total)")
    for i in range(nb):
        for j in range(nb):
            if norm[i, j] >= 0.005:
                ax.text(j, i, f"{norm[i, j] * 100:.0f}", ha="center", va="center",
                        color="white" if norm[i, j] < 0.6 else "black", fontsize=7)
    ax.plot([-.5, nb - .5], [-.5, nb - .5], color="red", lw=0.8, alpha=0.5,
            label="perfect calibration")
    ys = [i for i in range(nb) if present[i] and means[i] is not None]
    xs = [means[i] / bw - 0.5 for i in ys]
    ax.plot(xs, ys, "-o", color="orange", mec="black", mew=0.6, ms=7, lw=1.5,
            label="mean pred per true bin")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=2, fontsize=8, framealpha=0.9)
    bm = f"{balanced_mae:.3f}" if balanced_mae is not None else "n/a"
    ax.set_title(f"{title}\nexact-bin recall={diag:.3f}  +/-1bin={w1:.3f}  balanced MAE={bm}",
                 fontsize=9)
    fig.tight_layout()
    return fig
