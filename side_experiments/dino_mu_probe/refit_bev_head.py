"""Decoupled head refit: retrain a balanced 3-class readout on a collapsed
checkpoint's frozen BEV features.

Run: uv run python side_experiments/dino_mu_probe/refit_bev_head.py --dump <dump.npz>"""

import argparse

import numpy as np
from run_3class import SCHEMES, classes, report, tune_thresholds
from run_regression import bin_weights
from sklearn.linear_model import LogisticRegression, Ridge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True, help=".npz with bev_feat from extract_mu_predictions.py")
    ap.add_argument("--max-train", type=int, default=400_000)
    args = ap.parse_args()

    d = np.load(args.dump)
    tr, te = d["split"] == "train", d["split"] == "valid"
    X_tr, X_te = d["bev_feat"][tr].astype(np.float32), d["bev_feat"][te].astype(np.float32)
    mu_tr, mu_te = d["true_mu"][tr], d["true_mu"][te]
    if len(X_tr) > args.max_train:
        sub = np.random.default_rng(0).choice(len(X_tr), args.max_train, replace=False)
        X_tr, mu_tr = X_tr[sub], mu_tr[sub]
    print(f"{len(X_tr)} train / {len(X_te)} valid cells, feat dim {X_tr.shape[1]}")

    for scheme, (t1, t2) in SCHEMES.items():
        print(f"\n=== scheme {scheme}: untrav < {t1}, medium [{t1}, {t2}), trav >= {t2} ===")
        y_tr, y_te = classes(mu_tr, t1, t2), classes(mu_te, t1, t2)

        clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000).fit(X_tr, y_tr)
        report("BEV logistic refit (balanced)", y_te, clf.predict(X_te))

        reg = Ridge(alpha=10.0).fit(X_tr, mu_tr, sample_weight=bin_weights(mu_tr))
        pred_tr = np.clip(reg.predict(X_tr), 0, 1)
        pred_te = np.clip(reg.predict(X_te), 0, 1)
        a, b = tune_thresholds(pred_tr, y_tr)
        report(f"BEV ridge-bal refit, train-tuned thresholds ({a:.2f}, {b:.2f})", y_te, classes(pred_te, a, b))


if __name__ == "__main__":
    main()
