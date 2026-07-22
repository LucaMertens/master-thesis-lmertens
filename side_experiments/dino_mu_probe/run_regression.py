"""Continuous mu regression probe on frozen patch features.

Run: uv run python side_experiments/dino_mu_probe/run_regression.py"""

import argparse

import numpy as np
import yaml
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor

BIN_W = 0.1
ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0]


def l2_vector_normalize(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)


def bin_index_for_mu(mu):
    return np.clip((mu / BIN_W).astype(int), 0, int(1 / BIN_W) - 1)


def balanced_mae(mu_true, pred):
    """Mean of per-bin MAEs, so low-mu bins count as much as the high-mu mass."""
    # TODO: only include visible items
    err = np.abs(pred - mu_true)
    bins = bin_index_for_mu(mu_true)
    bin_indexes = np.unique(bins)
    bin_maes = [err[bins == b].mean() for b in bin_indexes]
    return float(np.mean(bin_maes))


def metrics(mu_true, pred):
    n = len(mu_true)
    sub = np.random.default_rng(0).choice(n, min(n, 4000), replace=False)
    dt = mu_true[sub, None] - mu_true[None, sub]
    dp = pred[sub, None] - pred[None, sub]
    comparable = dt != 0
    c_index = ((np.sign(dt) == np.sign(dp)) & comparable).sum() / comparable.sum()
    return {
        "mae": float(np.abs(pred - mu_true).mean()),
        "bal_mae": balanced_mae(mu_true, pred),
        "spearman": float(spearmanr(mu_true, pred).statistic),
        "c_index": float(c_index),
    }


def bin_weights(mu):
    """Inverse-frequency weight per mu bin, so rare low-mu bins count as much as
    the high-mu mass during fitting (the balanced analogue for training)."""
    bins = bin_index_for_mu(mu)
    counts = np.bincount(bins, minlength=int(1 / BIN_W))
    return 1.0 / counts[bins]


def pick_alpha(X, mu, mission, weighted):
    """Leave-one-train-mission-out CV, selecting alpha by balanced MAE."""
    best, best_score = ALPHAS[0], np.inf
    for alpha in ALPHAS:
        preds, trues = [], []
        for m in np.unique(mission):
            tr, te = mission != m, mission == m
            w = bin_weights(mu[tr]) if weighted else None
            reg = Ridge(alpha=alpha).fit(X[tr], mu[tr], sample_weight=w)
            preds.append(np.clip(reg.predict(X[te]), 0, 1))
            trues.append(mu[te])
        score = balanced_mae(np.concatenate(trues), np.concatenate(preds))
        if score < best_score:
            best, best_score = alpha, score
    return best, best_score


def eval_lomo(X, mu, mission, alpha):
    """Leave-one-mission-out for the balanced ridge. Pools all missions as training
    folds (like run_probe's LOMO), so slightly optimistic vs the official split."""
    print(f"\nLOMO, balanced ridge (a={alpha:g}):")
    print(
        f"  {'mission':<10} {'n':>5} {'n_low':>6} {'constBAL':>9} {'balMAE':>7} {'spear':>6} {'c-idx':>6}"
    )
    for m in np.unique(mission):
        te, tr = mission == m, mission != m
        reg = Ridge(alpha=alpha).fit(X[tr], mu[tr], sample_weight=bin_weights(mu[tr]))
        pred = np.clip(reg.predict(X[te]), 0, 1)
        r = metrics(mu[te], pred)
        const_bal = balanced_mae(mu[te], np.full(te.sum(), mu[tr].mean()))
        n_low = int((mu[te] <= 0.6).sum())
        print(
            f"  {m:<10} {te.sum():>5} {n_low:>6} {const_bal:>9.3f} {r['bal_mae']:>7.3f} "
            f"{r['spearman']:>6.3f} {r['c_index']:>6.3f}"
        )


def bin_table(mu_true, preds):
    names = list(preds)
    print(
        f"\n  {'bin':<10} {'n':>5}"
        + "".join(f" {n + ' pred':>16} {n + ' mae':>12}" for n in names)
    )
    bins = bin_index_for_mu(mu_true)
    for b in np.unique(bins):
        sel = bins == b
        row = f"  {b * BIN_W:.1f}-{(b + 1) * BIN_W:.1f}   {sel.sum():>5}"
        for n in names:
            p = preds[n][sel]
            row += f" {p.mean():>16.3f} {np.abs(p - mu_true[sel]).mean():>12.3f}"
        print(row)


def plot_scatter(mu_tr, pred_tr, mu_te, pred_te, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    for ax, mu, pred, title in [
        (axes[0], mu_tr, pred_tr, "train (fit on this)"),
        (axes[1], mu_te, pred_te, "valid (held out)"),
    ]:
        ax.scatter(mu, pred, s=4, alpha=0.3)
        centers, means = [], []
        bins = bin_index_for_mu(mu)
        for b in np.unique(bins):
            centers.append((b + 0.5) * BIN_W)
            means.append(pred[bins == b].mean())
        ax.plot(centers, means, "o-", color="red", label="bin mean")
        ax.plot([0, 1], [0, 1], "--", color="grey")
        ax.set_xlabel("true mu")
        ax.set_title(title)
        ax.legend()
    axes[0].set_ylabel("predicted mu")
    fig.suptitle("Ridge regression on L2-normalized frozen patch features")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close("all")
    print("\nscatter ->", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--features", default="side_experiments/dino_mu_probe/output/features_reg.npz"
    )
    ap.add_argument(
        "--cfg", default="training/traversability_network/configs/grandtour_lite.yaml"
    )
    ap.add_argument(
        "--plot", default="side_experiments/dino_mu_probe/output/regression_scatter.png"
    )
    ap.add_argument(
        "--lomo", action="store_true", help="also run leave-one-mission-out"
    )
    args = ap.parse_args()

    d = np.load(args.features)
    X, mu, mission = (
        l2_vector_normalize(d["features"].astype(np.float64)),
        d["mu"],
        d["mission"],
    )
    cfg = yaml.safe_load(open(args.cfg))
    tr = np.isin(mission, cfg["DATASET"]["TRAIN_MISSIONS"])
    te = np.isin(mission, cfg["DATASET"]["VALID_MISSIONS"])
    print(
        f"model {d['model']}, {tr.sum()} train / {te.sum()} valid points, "
        f"mid-band n={int(((mu > 0.6) & (mu < 0.85)).sum())}"
    )

    results = {}
    const = np.full(te.sum(), mu[tr].mean())
    results["constant"] = metrics(mu[te], const)

    preds = {"const": const}
    for name, weighted in [("ridge", False), ("ridge-bal", True)]:
        alpha, cv = pick_alpha(X[tr], mu[tr], mission[tr], weighted)
        w = bin_weights(mu[tr]) if weighted else None
        reg = Ridge(alpha=alpha).fit(X[tr], mu[tr], sample_weight=w)
        pred = np.clip(reg.predict(X[te]), 0, 1)
        results[f"{name} (a={alpha:g}, cv {cv:.3f})"] = metrics(mu[te], pred)
        preds[name] = pred
        if name == "ridge-bal":
            pred_tr = np.clip(reg.predict(X[tr]), 0, 1)

    knn = KNeighborsRegressor(n_neighbors=20).fit(X[tr], mu[tr])
    pred = np.clip(knn.predict(X[te]), 0, 1)
    results["knn20"] = metrics(mu[te], pred)
    preds["knn20"] = pred

    print(f"\n  {'method':<28} {'MAE':>6} {'balMAE':>7} {'spear':>6} {'c-idx':>6}")
    for name, m in results.items():
        print(
            f"  {name:<28} {m['mae']:>6.3f} {m['bal_mae']:>7.3f} {m['spearman']:>6.3f} {m['c_index']:>6.3f}"
        )
    print(
        "\n  network reference: valid/balanced_mae 0.301 (BMC best) / 0.320 (frozen-DINO ep0), both collapsed"
    )

    bin_table(mu[te], preds)
    if args.lomo:
        eval_lomo(X, mu, mission, alpha)
    plot_scatter(mu[tr], pred_tr, mu[te], preds["ridge-bal"], args.plot)


if __name__ == "__main__":
    main()
