"""Three-class (untraversable / medium / traversable) traversability probe.

Run: uv run python side_experiments/dino_mu_probe/run_3class.py"""

import argparse
import os

import numpy as np
import yaml
from run_regression import bin_weights, l2_vector_normalize
from sklearn.linear_model import LogisticRegression, Ridge

SCHEMES = {"0.3/0.6": (0.3, 0.6), "0.6/0.85": (0.6, 0.85)}
CLASS_NAMES = ["untrav", "medium", "trav"]


def classes(mu, t1, t2):
    return np.digitize(mu, [t1, t2])


def confusion(y_true, y_pred):
    cm = np.zeros((3, 3), dtype=int)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def balanced_acc(cm):
    seen = cm.sum(1) > 0
    return float((np.diag(cm)[seen] / cm.sum(1)[seen]).mean())


def report(name, y_true, y_pred):
    cm = confusion(y_true, y_pred)
    recalls = np.divide(
        np.diag(cm), cm.sum(1), out=np.full(3, np.nan), where=cm.sum(1) > 0
    )
    print(f"\n  {name}: balanced acc {balanced_acc(cm):.3f} (chance 0.333)")
    print(
        f"    {'true \\ pred':<14}"
        + "".join(f"{c:>9}" for c in CLASS_NAMES)
        + f"{'n':>8} {'recall':>8}"
    )
    for i, c in enumerate(CLASS_NAMES):
        row = cm[i] / max(cm[i].sum(), 1)
        print(
            f"    {c:<14}"
            + "".join(f"{v:>9.2f}" for v in row)
            + f"{cm[i].sum():>8} {recalls[i]:>8.2f}"
        )
    return balanced_acc(cm)


def tune_thresholds(pred_mu, y_true):
    """Best (t1, t2) by balanced accuracy, via per-class prediction CDFs on a grid."""
    grid = np.arange(0.02, 1.0, 0.01)
    F = np.stack([
        np.searchsorted(np.sort(pred_mu[y_true == c]), grid)
        / max((y_true == c).sum(), 1)
        for c in range(3)
    ])
    r0, r1 = F[0][:, None], F[1][None, :] - F[1][:, None]
    r2 = 1 - F[2][None, :]
    seen = np.array([(y_true == c).any() for c in range(3)])
    bal = (r0 * seen[0] + r1 * seen[1] + r2 * seen[2]) / seen.sum()
    bal = np.where(grid[:, None] < grid[None, :], bal, -1)
    i, j = np.unravel_index(bal.argmax(), bal.shape)
    return grid[i], grid[j]


def probe_features(args, t1, t2):
    d = np.load(args.features)
    X, mu, mission = (
        l2_vector_normalize(d["features"].astype(np.float64)),
        d["mu"],
        d["mission"],
    )
    cfg = yaml.safe_load(open(args.cfg))
    tr = np.isin(mission, cfg["DATASET"]["TRAIN_MISSIONS"])
    te = np.isin(mission, cfg["DATASET"]["VALID_MISSIONS"])
    y_tr, y_te = classes(mu[tr], t1, t2), classes(mu[te], t1, t2)

    best, best_score = None, -1
    for C in [0.01, 0.1, 1.0, 10.0]:
        scores = []
        for m in np.unique(mission[tr]):
            fold = mission[tr] != m
            if len(np.unique(y_tr[fold])) < 2 or not (~fold).any():
                continue
            clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000).fit(
                X[tr][fold], y_tr[fold]
            )
            scores.append(
                balanced_acc(confusion(y_tr[~fold], clf.predict(X[tr][~fold])))
            )
        if scores and np.mean(scores) > best_score:
            best, best_score = C, float(np.mean(scores))
    clf = LogisticRegression(C=best, class_weight="balanced", max_iter=5000).fit(
        X[tr], y_tr
    )
    logistic = report(
        f"DINO logistic probe (C={best:g}, cv {best_score:.3f})",
        y_te,
        clf.predict(X[te]),
    )

    reg = Ridge(alpha=10.0).fit(X[tr], mu[tr], sample_weight=bin_weights(mu[tr]))
    pred = np.clip(reg.predict(X[te]), 0, 1)
    report("DINO ridge-bal, fixed thresholds", y_te, classes(pred, t1, t2))
    return logistic


def net_predictions(path, t1, t2):
    d = np.load(path)
    tr, te = d["split"] == "train", d["split"] == "valid"
    y_te = classes(d["true_mu"][te], t1, t2)
    name = os.path.basename(path).replace(".npz", "")
    report(f"net {name}, fixed thresholds", y_te, classes(d["pred_mu"][te], t1, t2))
    a, b = tune_thresholds(d["pred_mu"][tr], classes(d["true_mu"][tr], t1, t2))
    report(
        f"net {name}, tuned thresholds ({a:.2f}, {b:.2f})",
        y_te,
        classes(d["pred_mu"][te], a, b),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--features", default="side_experiments/dino_mu_probe/output/features_reg.npz"
    )
    ap.add_argument(
        "--cfg", default="training/traversability_network/configs/grandtour_lite.yaml"
    )
    ap.add_argument(
        "--net",
        nargs="*",
        default=[],
        help="prediction .npz files from extract_mu_predictions.py",
    )
    ap.add_argument("--skip-probe", action="store_true", help="only score --net files")
    args = ap.parse_args()

    for scheme, (t1, t2) in SCHEMES.items():
        print(
            f"\n=== scheme {scheme}: untrav < {t1}, medium [{t1}, {t2}), trav >= {t2} ==="
        )
        if not args.skip_probe:
            probe_features(args, t1, t2)
        for path in args.net:
            net_predictions(path, t1, t2)


if __name__ == "__main__":
    main()
