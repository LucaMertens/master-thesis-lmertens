"""Ceiling probe, step 2: leave-one-mission-out linear probe on cached DINO features.
Answers: is low mu separable from appearance at all, across missions?"""

import argparse

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score

MIN_POS = 10  # skip eval folds with fewer held-out low-mu points


def fit_probe(X_train, y_train):
    """Train a binary classifier: low-mu (y=1) vs high-mu (y=0) patch features.
    X_train is standardized (zero mean, unit variance per dim). Must return an
    object exposing predict_proba or decision_function."""
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000).fit(X_train, y_train)


def scores(clf, X):
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)[:, 1]
    return clf.decision_function(X)


def plot_pca(X, mu, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = PCA(2).fit_transform(X)
    sc = plt.scatter(p[:, 0], p[:, 1], c=mu, s=4, cmap="viridis")
    plt.colorbar(sc, label="mu")
    plt.title("frozen DINO patch features at path points (PCA)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    print("PCA plot ->", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="side_experiments/dino_mu_probe/output/features.npz")
    ap.add_argument("--plot", default="side_experiments/dino_mu_probe/output/pca.png")
    args = ap.parse_args()

    d = np.load(args.features)
    X, y, mission, mu = d["features"], d["y"], d["mission"], d["mu"]
    print(f"{len(y)} points, {int(y.sum())} low-mu, model {d['model']}")

    rows = []
    for m in np.unique(mission):
        te = mission == m
        if y[te].sum() < MIN_POS:
            continue
        tr = ~te
        mean, std = X[tr].mean(0), X[tr].std(0) + 1e-8
        clf = fit_probe((X[tr] - mean) / std, y[tr])
        s = scores(clf, (X[te] - mean) / std)
        rows.append((m, roc_auc_score(y[te], s), average_precision_score(y[te], s),
                     int(y[te].sum()), int(te.sum())))

    print(f"\n{'mission':<10} {'AUROC':>6} {'AP':>6} {'AP-base':>8} {'n_low':>6} {'n':>6}")
    for m, auc, ap_score, n_pos, n in rows:
        print(f"{m:<10} {auc:>6.3f} {ap_score:>6.3f} {n_pos / n:>8.3f} {n_pos:>6} {n:>6}")
    print(f"\nmean AUROC {np.mean([r[1] for r in rows]):.3f}  "
          "(~0.5: mu not predictable from appearance; >~0.75: pipeline problem, not a data ceiling)")

    plot_pca(X, mu, args.plot)


if __name__ == "__main__":
    main()
