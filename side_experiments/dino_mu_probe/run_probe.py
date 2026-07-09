"""Ceiling probe, step 2: leave-one-mission-out linear probe on cached DINO features.
Answers: is low mu separable from appearance at all, across missions?"""

import argparse

import numpy as np
import yaml
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score

MIN_POS = 10  # skip eval folds with fewer held-out low-mu points


def derive_labels(mu, mu_low, mu_high):
    """Binary label from a configurable threshold pair, applied to the continuous
    mu already cached in features.npz. features.npz itself only contains points
    with mu<=0.6 or mu>=0.85 (extract_features.py's OVERSAMPLE_MU_MAX/HIGH_MU), so
    narrowing the thresholds just relabels/drops already-cached points, while
    widening them past [0.6, 0.85) cannot recover points nobody extracted -- rerun
    extract_features.py with different constants for that."""
    keep = (mu <= mu_low) | (mu >= mu_high)
    y = (mu <= mu_low).astype(np.int64)
    return keep, y


def concordance_index(y_true, y_pred):
    """Generalization of AUROC to a continuous target: fraction of all comparable
    pairs where the prediction orders them the same way the true mu does. Reduces
    exactly to AUROC when y_true is binary, so it is directly comparable to the
    classifier numbers above without picking any low/high threshold at all."""
    dt = y_true[:, None] - y_true[None, :]
    dp = y_pred[:, None] - y_pred[None, :]
    comparable = dt != 0
    concordant = (np.sign(dt) == np.sign(dp)) & comparable
    return concordant.sum() / comparable.sum()


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
    plt.close("all")
    print("PCA plot ->", path)


def plot_pca3d(X, mu, path):
    """Multi-angle 3-component PCA: a single static 3D view hides separation just
    as easily as 2D does, so render several azimuths of the same embedding."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = PCA(3).fit_transform(X)
    fig = plt.figure(figsize=(14, 12))
    for i, (elev, azim) in enumerate([(20, 30), (20, 120), (20, 210), (20, 300), (70, 30), (0, 0)]):
        ax = fig.add_subplot(2, 3, i + 1, projection="3d")
        sc = ax.scatter(p[:, 0], p[:, 1], p[:, 2], c=mu, s=3, cmap="viridis")
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"elev={elev} azim={azim}", fontsize=9)
    fig.colorbar(sc, ax=fig.axes, label="mu", shrink=0.6)
    fig.suptitle("frozen DINO patch features at path points (3-component PCA)")
    plt.savefig(path, dpi=150)
    plt.close("all")
    print("3D PCA plot ->", path)


def fit_official_split(X, y, mu, mission, train_missions, valid_missions):
    """Train only on the network's real TRAIN_MISSIONS, test only on its real
    VALID_MISSIONS (pooled). Unlike the LOMO loop below, this never lets other
    held-out missions leak into the training fold, so it is the honest analogue
    of what the network itself has to generalize across. Returns everything
    downstream plots/metrics need, or None if there is not enough data."""
    tr = np.isin(mission, train_missions)
    te = np.isin(mission, valid_missions)
    if tr.sum() == 0 or te.sum() == 0 or y[te].sum() < MIN_POS:
        print("official train->valid split: not enough data in features.npz, skipping "
              "(rerun extract_features.py without --missions to include all TRAIN+VALID)")
        return None
    mean, std = X[tr].mean(0), X[tr].std(0) + 1e-8
    Xtr_n, Xte_n = (X[tr] - mean) / std, (X[te] - mean) / std
    clf = fit_probe(Xtr_n, y[tr])
    s = scores(clf, Xte_n)
    return {"tr": tr, "te": te, "Xtr_n": Xtr_n, "Xte_n": Xte_n, "y_tr": y[tr], "y_te": y[te], "s": s,
            "mu_tr": mu[tr], "mu_te": mu[te], "clf": clf, "mission_te": mission[te],
            "train_missions": sorted(set(mission[tr])), "valid_missions": sorted(set(mission[te]))}


def eval_official_split(fit):
    auc = roc_auc_score(fit["y_te"], fit["s"])
    ap_score = average_precision_score(fit["y_te"], fit["s"])
    print(f"\nofficial train->valid split (train={fit['train_missions']}, "
          f"valid={fit['valid_missions']})")
    print(f"AUROC {auc:.3f}  AP {ap_score:.3f}  AP-base {fit['y_te'].mean():.3f}  "
          f"n_low {int(fit['y_te'].sum())}  n {int(fit['te'].sum())}")

    print("  per-valid-mission (same train-only fit, honest, no leakage):")
    for m in fit["valid_missions"]:
        sel = fit["mission_te"] == m
        y_m, s_m = fit["y_te"][sel], fit["s"][sel]
        if y_m.sum() < 5:
            print(f"    {m:<10} n_low={int(y_m.sum())} (too few for AUROC)")
            continue
        print(f"    {m:<10} AUROC {roc_auc_score(y_m, s_m):.3f}  n_low={int(y_m.sum())}  n={int(sel.sum())}")
    return auc


def plot_score_hist(fit, auc, path):
    """1D view of exactly what AUROC integrates over: the classifier's P(low-mu)
    on held-out VALID missions, split by true label. Unlike PCA this is a
    supervised projection (the probe's own decision axis), so it is the most
    direct way to see the separation the AUROC number describes."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s, y = fit["s"], fit["y_te"]
    bins = np.linspace(0, 1, 41).tolist()
    plt.hist(s[y == 0], bins=bins, alpha=0.6, density=True, label=f"high-mu (n={int((y==0).sum())})")
    plt.hist(s[y == 1], bins=bins, alpha=0.6, density=True, label=f"low-mu (n={int((y==1).sum())})")
    plt.xlabel("probe P(low-mu)")
    plt.ylabel("density")
    plt.title(f"official valid-split probe scores, AUROC={auc:.3f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close("all")
    print("score histogram ->", path)


def plot_classifier_axis(fit, path):
    """x-axis = raw logit along the FITTED classifier's own weight direction (the
    exact axis AUROC is computed on); y-axis = PC1, an unsupervised direction
    orthogonal-ish to it, for visual spread. Train panel shows the separation the
    classifier was fit to produce; valid panel shows how much of it survives on
    missions it never saw, which is the honest 0.795 story."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pc1 = PCA(n_components=1).fit(fit["Xtr_n"])
    x_tr = fit["clf"].decision_function(fit["Xtr_n"])
    x_te = fit["clf"].decision_function(fit["Xte_n"])
    y_tr = pc1.transform(fit["Xtr_n"])[:, 0]
    y_te = pc1.transform(fit["Xte_n"])[:, 0]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    axes[0].scatter(x_tr, y_tr, c=fit["mu_tr"], s=4, cmap="viridis")
    sc = axes[1].scatter(x_te, y_te, c=fit["mu_te"], s=4, cmap="viridis")
    for ax, title in zip(axes, ["train (fit on this)", "valid (held out)"]):
        ax.axvline(0, color="grey", lw=0.8, ls="--")
        ax.set_xlabel("classifier direction (logit, >0 -> predicts low-mu)")
        ax.set_title(title)
    axes[0].set_ylabel("PC1 (unsupervised)")
    fig.colorbar(sc, ax=axes, label="mu", shrink=0.8)
    fig.suptitle("DINO features projected onto the fitted low-mu classifier direction")
    plt.savefig(path, dpi=150)
    plt.close("all")
    print("classifier-axis plot ->", path)


def plot_tsne(X, mu, path, perplexity=30, seed=0):
    """Nonlinear complement to PCA: t-SNE preserves local neighborhoods rather
    than global variance, so it can reveal low-mu clustering that a linear
    projection spreads across low-variance directions and hides."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    Xs = (X - X.mean(0)) / (X.std(0) + 1e-8)
    emb = TSNE(n_components=2, init="pca", perplexity=perplexity, random_state=seed).fit_transform(Xs)
    sc = plt.scatter(emb[:, 0], emb[:, 1], c=mu, s=4, cmap="viridis")
    plt.colorbar(sc, label="mu")
    plt.title("frozen DINO patch features at path points (t-SNE)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close("all")
    print("t-SNE plot ->", path)


def knn_purity(fit, k=20):
    """Local, projection-free separability check: among each held-out point's k
    nearest neighbors in raw (standardized) 768-d feature space -- found within
    the TRAIN missions only -- what fraction are low-mu? Comparing against the
    train base rate is confounded by train/valid domain shift (valid points can
    sit in a sparser part of feature space for reasons unrelated to mu), so the
    fair comparison is the two class-conditional means against EACH OTHER, and
    treating the neighbor low-mu fraction itself as a nonparametric classifier
    score gives a kNN AUROC directly comparable to the logistic-regression one."""
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=k).fit(fit["Xtr_n"])
    _, idx = nn.kneighbors(fit["Xte_n"])
    neighbor_low_frac = fit["y_tr"][idx].mean(1)
    y_te = fit["y_te"]
    knn_auc = roc_auc_score(y_te, neighbor_low_frac)
    print(f"\nk-NN local structure (k={k}, train base rate {fit['y_tr'].mean():.3f}):")
    print(f"  neighbors of true low-mu valid points:  mean low-mu fraction {neighbor_low_frac[y_te == 1].mean():.3f}")
    print(f"  neighbors of true high-mu valid points: mean low-mu fraction {neighbor_low_frac[y_te == 0].mean():.3f}")
    print(f"  kNN-score AUROC (nonparametric, complements the logistic probe): {knn_auc:.3f}")


def eval_regression_official_split(X, mu, mission, train_missions, valid_missions):
    """Regression equivalent of the binary probe: predict continuous mu directly
    with Ridge, no low/high threshold anywhere. Scored with the concordance index
    (the AUROC generalization above) plus MAE, on the same leakage-free
    train-only -> valid-only split as eval_official_split."""
    from sklearn.linear_model import Ridge

    tr = np.isin(mission, train_missions)
    te = np.isin(mission, valid_missions)
    mean, std = X[tr].mean(0), X[tr].std(0) + 1e-8
    Xtr_n, Xte_n = (X[tr] - mean) / std, (X[te] - mean) / std
    reg = Ridge(alpha=10.0).fit(Xtr_n, mu[tr])
    pred = reg.predict(Xte_n)
    mu_te = mu[te]
    c_index = concordance_index(mu_te, pred)
    mae = np.mean(np.abs(pred - mu_te))
    print("\nregression equivalent (Ridge on continuous mu, no threshold, same split):")
    print(f"  MAE {mae:.3f}  concordance-index {c_index:.3f}  "
          "(same formula as AUROC, generalized to a continuous target)")
    return reg, pred, mu_te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="side_experiments/dino_mu_probe/output/features.npz")
    ap.add_argument("--plot", default="side_experiments/dino_mu_probe/output/pca.png")
    ap.add_argument("--plot3d", default="side_experiments/dino_mu_probe/output/pca3d.png")
    ap.add_argument("--tsne", default="side_experiments/dino_mu_probe/output/tsne.png")
    ap.add_argument("--score-hist", default="side_experiments/dino_mu_probe/output/score_hist.png")
    ap.add_argument("--classifier-axis", default="side_experiments/dino_mu_probe/output/classifier_axis.png")
    ap.add_argument("--cfg", default="training/traversability_network/configs/grandtour_lite.yaml")
    ap.add_argument("--mu-low", type=float, default=0.6, help="mu <= this -> low-mu label")
    ap.add_argument("--mu-high", type=float, default=0.85, help="mu >= this -> high-mu label")
    args = ap.parse_args()

    d = np.load(args.features)
    X, mission, mu = d["features"], d["mission"], d["mu"]
    keep, y = derive_labels(mu, args.mu_low, args.mu_high)
    if args.mu_low > 0.6 or args.mu_high < 0.85:
        print("NOTE: features.npz was extracted with only mu<=0.6 or mu>=0.85 cached; "
              "--mu-low/--mu-high outside that range only relabels/drops cached points, "
              "it cannot add points from the un-extracted [0.6, 0.85) band.")
    if not keep.all():
        print(f"dropping {(~keep).sum()} points outside [mu<={args.mu_low} or mu>={args.mu_high}]")
        X, mission, mu, y = X[keep], mission[keep], mu[keep], y[keep]
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
    print("NOTE: this LOMO loop pools TRAIN_MISSIONS + VALID_MISSIONS as candidate folds, so "
          "when a VALID mission is held out, OTHER valid missions still leak into its training "
          "fold, unlike the real network. See 'official train->valid split' below for the "
          "leakage-free number.")

    cfg = yaml.safe_load(open(args.cfg))
    fit = fit_official_split(X, y, mu, mission, cfg["DATASET"]["TRAIN_MISSIONS"], cfg["DATASET"]["VALID_MISSIONS"])
    if fit is not None:
        auc = eval_official_split(fit)
        plot_score_hist(fit, auc, args.score_hist)
        plot_classifier_axis(fit, args.classifier_axis)
        knn_purity(fit)

    eval_regression_official_split(X, mu, mission, cfg["DATASET"]["TRAIN_MISSIONS"], cfg["DATASET"]["VALID_MISSIONS"])

    plot_pca(X, mu, args.plot)
    plot_pca3d(X, mu, args.plot3d)
    plot_tsne(X, mu, args.tsne)


if __name__ == "__main__":
    main()
