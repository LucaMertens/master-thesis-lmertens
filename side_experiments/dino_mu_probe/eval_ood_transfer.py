"""Ceiling probe, step 3: score the frozen mu probe on the OOD tiers, zero-shot.

Run: uv run python side_experiments/dino_mu_probe/eval_ood_transfer.py --ood-features <feats.npz>"""

import argparse

import numpy as np
import yaml
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, roc_auc_score

from run_probe import MIN_POS, concordance_index, fit_probe, scores

TIERS = {
    "near-forest": ["KAEB-1", "ROOT-1"],
    "near-gravel": ["EIG-1", "EIG-2", "PIL-2"],
    "far-snow": ["SNOW-1", "SNOW-2", "SNOW-3"],
    "far-arc": ["ARC-1", "ARC-2", "ARC-3", "ARC-4"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id-features", default="side_experiments/dino_mu_probe/output/features.npz")
    ap.add_argument("--ood-features", required=True)
    ap.add_argument("--cfg", default="training/traversability_network/configs/grandtour_lite.yaml")
    ap.add_argument("--mu-low", type=float, default=0.6)
    ap.add_argument("--mu-high", type=float, default=0.85)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.cfg))
    train_missions = cfg["DATASET"]["TRAIN_MISSIONS"]

    id_d = np.load(args.id_features)
    ood_d = np.load(args.ood_features)
    assert str(id_d["model"]) == str(ood_d["model"]), \
        f"model mismatch: id={id_d['model']} ood={ood_d['model']}"
    model_name = str(id_d["model"])

    tr = np.isin(id_d["mission"], train_missions)
    X_tr, mu_tr = id_d["features"][tr], id_d["mu"][tr]
    keep = (mu_tr <= args.mu_low) | (mu_tr >= args.mu_high)
    X_tr, mu_tr = X_tr[keep], mu_tr[keep]
    y_tr = (mu_tr <= args.mu_low).astype(np.int64)

    mean, std = X_tr.mean(0), X_tr.std(0) + 1e-8
    Xtr_n = (X_tr - mean) / std
    clf = fit_probe(Xtr_n, y_tr)
    reg = Ridge(alpha=10.0).fit(Xtr_n, mu_tr)

    X_ood, mu_ood, mission_ood = ood_d["features"], ood_d["mu"], ood_d["mission"]
    keep = (mu_ood <= args.mu_low) | (mu_ood >= args.mu_high)
    X_ood, mu_ood, mission_ood = X_ood[keep], mu_ood[keep], mission_ood[keep]
    y_ood = (mu_ood <= args.mu_low).astype(np.int64)

    print(f"model {model_name}, fit on {len(y_tr)} TRAIN_MISSIONS points ({int(y_tr.sum())} low-mu)\n")
    print(f"{'tier':<14}{'AUROC':>7}{'AP':>7}{'AP-base':>9}{'n_low':>7}{'n':>7}{'MAE':>8}{'concord':>9}")
    for tier, missions in TIERS.items():
        sel = np.isin(mission_ood, missions)
        if sel.sum() == 0:
            print(f"{tier:<14}  no data extracted for {missions}")
            continue
        Xt_n = (X_ood[sel] - mean) / std
        y_t, mu_t, mission_t = y_ood[sel], mu_ood[sel], mission_ood[sel]
        pred_mu = reg.predict(Xt_n)
        mae = np.mean(np.abs(pred_mu - mu_t))
        c_index = concordance_index(mu_t, pred_mu)

        if y_t.sum() < MIN_POS or (y_t == 0).sum() < MIN_POS:
            print(f"{tier:<14}{'--':>7}{'--':>7}{y_t.mean():>9.3f}{int(y_t.sum()):>7}"
                  f"{int(sel.sum()):>7}{mae:>8.3f}{c_index:>9.3f}  (too few for AUROC)")
            continue
        s = scores(clf, Xt_n)
        auc = roc_auc_score(y_t, s)
        ap = average_precision_score(y_t, s)
        print(f"{tier:<14}{auc:>7.3f}{ap:>7.3f}{y_t.mean():>9.3f}{int(y_t.sum()):>7}"
              f"{int(sel.sum()):>7}{mae:>8.3f}{c_index:>9.3f}")

        for m in missions:
            msel = mission_t == m
            if msel.sum() == 0:
                continue
            y_m = y_t[msel]
            if y_m.sum() < 5 or (y_m == 0).sum() < 5:
                print(f"    {m:<10} n_low={int(y_m.sum())} n={int(msel.sum())} (too few for per-mission AUROC)")
                continue
            print(f"    {m:<10} AUROC {roc_auc_score(y_m, s[msel]):.3f} "
                  f"n_low={int(y_m.sum())} n={int(msel.sum())}")


if __name__ == "__main__":
    main()
