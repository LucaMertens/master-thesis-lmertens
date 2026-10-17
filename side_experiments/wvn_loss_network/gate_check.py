"""Fit and sanity-check the Mahalanobis OOD gate on DINO features."""

import argparse
import os

import cv2
import numpy as np
import yaml
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score

QUANTILES = [0.90, 0.95, 0.99]
MIN_LOW = 50  # lows needed for a per-mission AUROC estimate
OVERLAY_MISSIONS = ["ALB-3", "LMB-2", "KAEB-1", "EIG-1", "ARC-1", "SNOW-1"]


def l2n(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


class MahalanobisGate:
    """Mean + Ledoit-Wolf precision on L2-normalized features (no per-dim standardization)."""

    def fit(self, X):
        self.mean = X.mean(0)
        self.prec = LedoitWolf().fit(X - self.mean).precision_.astype(np.float64)
        return self

    def score(self, X):
        d = (X - self.mean).astype(np.float64)
        return np.einsum("ij,jk,ik->i", d, self.prec, d)


def auroc(scores, y, pos, neg):
    sel = np.isin(y, pos) | np.isin(y, neg)
    if np.isin(y[sel], pos).sum() == 0 or np.isin(y[sel], neg).sum() == 0:
        return float("nan")
    return roc_auc_score(np.isin(y[sel], pos), scores[sel])


def eval_fold(name, gate, fit_scores, feats, y, sel):
    s, yy = gate.score(feats[sel]), y[sel]
    taus = np.quantile(fit_scores, QUANTILES)
    fpr = [(s[yy == 0] > t).mean() if (yy == 0).any() else float("nan") for t in taus]
    print(
        f"  {name:22s} n={sel.sum():5d}  "
        f"AUROC low|high {auroc(s, yy, [1], [0]):.3f}  low|rest {auroc(s, yy, [1], [0, -1]):.3f}  "
        f"high-mu fire-rate @q90/95/99 {fpr[0]:.2f}/{fpr[1]:.2f}/{fpr[2]:.2f}  "
        f"mean score {s.mean():.0f} (fit {fit_scores.mean():.0f})"
    )


def overlay_frames(cfg, gate, lo, hi, out_dir, n_frames=4):
    ds, cache_root = cfg["DATASET"], cfg["DATASET"]["DINO_CACHE_ROOT"]
    os.makedirs(out_dir, exist_ok=True)
    for mission in OVERLAY_MISSIONS:
        cache_dir = os.path.join(cache_root, mission)
        if not os.path.exists(os.path.join(cache_dir, "meta.npz")):
            print(f"  {mission}: no DINO cache, skipping overlay")
            continue
        feats = np.load(os.path.join(cache_dir, "features.npy"), mmap_mode="r")
        fc = np.load(os.path.join(ds["DATA_ROOT"], mission, "image_manifest.npz"))["front_center"]
        order = np.argsort(fc["ts_ns"].astype(np.int64))  # cache row order
        root = os.path.join(ds["RESIZED_ROOT"], mission)
        if not os.path.isdir(root):
            root = os.path.join(ds["DATA_ROOT"], mission)
        rows = np.linspace(0.15, 0.85, n_frames) * (len(order) - 1)
        for r in rows.astype(int):
            f = np.asarray(feats[r], np.float32)  # (D, h, w)
            D, h, w = f.shape
            s = gate.score(l2n(f.reshape(D, -1).T)).reshape(h, w)
            img = cv2.imread(os.path.join(root, fc["path"][order[r]]))
            assert img is not None
            heat = np.clip((s - lo) / (hi - lo), 0, 1)
            heat = cv2.resize((heat * 255).astype(np.uint8), (img.shape[1], img.shape[0]),
                              interpolation=cv2.INTER_LINEAR)
            blend = cv2.addWeighted(img, 0.5, cv2.applyColorMap(heat, cv2.COLORMAP_JET), 0.5, 0)
            cv2.imwrite(os.path.join(out_dir, f"{mission}_{r:05d}.jpg"), blend)
        print(f"  {mission}: {n_frames} overlays written")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="training/traversability_network/configs/grandtour_dino.yaml")
    ap.add_argument("--features", default="side_experiments/dino_mu_probe/output/features_reg.npz")
    ap.add_argument("--out", default="side_experiments/wvn_loss_network/output")
    ap.add_argument("--no-overlays", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.cfg))
    train_missions = cfg["DATASET"]["TRAIN_MISSIONS"]
    d = np.load(args.features, allow_pickle=True)
    feats, y, mission = l2n(d["features"].astype(np.float32)), d["y"], d["mission"]
    print(f"{len(y)} points, model {d['model']}")

    for fit_pop, fit_name in [(None, "all traversed"), (0, "high-mu only")]:
        print(f"\n== gate fit on {fit_name} (train missions) ==")
        fit_sel = np.isin(mission, train_missions)
        if fit_pop is not None:
            fit_sel &= y == fit_pop
        gate = MahalanobisGate().fit(feats[fit_sel])
        fit_scores = gate.score(feats[fit_sel])

        valid = [m for m in np.unique(mission) if m not in train_missions]
        for m in valid:
            eval_fold(f"valid {m}", gate, fit_scores, feats, y, mission == m)
        eval_fold("valid pooled", gate, fit_scores, feats, y, np.isin(mission, valid))

        print("  -- leave-one-mission-out on train missions --")
        for m in train_missions:
            held = mission == m
            if (y[held] == 1).sum() < MIN_LOW:
                continue
            lomo_sel = np.isin(mission, train_missions) & ~held
            if fit_pop is not None:
                lomo_sel &= y == fit_pop
            g = MahalanobisGate().fit(feats[lomo_sel])
            eval_fold(f"lomo {m}", g, g.score(feats[lomo_sel]), feats, y, held)

    # deployment gate on all train missions, with per-mission tau quantiles (global tau misfires under mission shift)
    gate = MahalanobisGate().fit(feats[np.isin(mission, train_missions)])
    fit_scores = gate.score(feats[np.isin(mission, train_missions)])
    all_missions = sorted(np.unique(mission).tolist())
    mission_tau = np.stack([
        np.quantile(gate.score(feats[mission == m]), QUANTILES) for m in all_missions
    ])
    os.makedirs(args.out, exist_ok=True)
    np.savez(
        os.path.join(args.out, "gate.npz"),
        mean=gate.mean, precision=gate.prec, model=str(d["model"]),
        quantile_q=np.array(QUANTILES), quantile_tau=np.quantile(fit_scores, QUANTILES),
        mission=np.array(all_missions), mission_tau=mission_tau,
        fit_scores=fit_scores,
    )
    print(f"\ngate saved -> {args.out}/gate.npz")

    if not args.no_overlays:
        print("\n== per-patch score overlays ==")
        lo, hi = np.quantile(fit_scores, [0.5, 0.999])
        overlay_frames(cfg, gate, lo, hi, os.path.join(args.out, "overlays"))


if __name__ == "__main__":
    main()
