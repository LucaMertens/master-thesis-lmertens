#!/usr/bin/env python
"""Align WayFASTER runs into one frame via DINO place matches and 1-point SE(2) RANSAC."""

import argparse
import pathlib

import cv2
import numpy as np
import pandas as pd
import torch
from shared_utils.paths import DATALOCAL_ROOT, WAYFASTER_DATASET_ROOT
from ood_detection.common.dino import dino_pooled_features, load_dino, patch_size
from scipy.spatial.transform import Rotation as Rot
from torch.utils.data import DataLoader, Dataset
from common import RUNS_2023_06_28, parse_vec

def load_run(cam: str, split: str, ts_folder: str, root: pathlib.Path = WAYFASTER_DATASET_ROOT) -> dict:
    """Image paths with odometry pose interpolated onto the image timestamps."""
    run_dir = root / cam / f"data_{split}" / ts_folder
    im = pd.read_csv(run_dir / "images.csv", usecols=["timestamp", "image"]).sort_values("timestamp")
    st = pd.read_csv(run_dir / "states.csv", usecols=["timestamp", "position", "quaternion"]).sort_values("timestamp")

    ts = st["timestamp"].to_numpy()
    P = np.stack([parse_vec(x) for x in st["position"]])[:, :2]
    Q = np.stack([parse_vec(x) for x in st["quaternion"]])[:, [1, 2, 3, 0]]  # wxyz -> xyzw
    yaw = Rot.from_quat(Q).as_euler("ZYX")[:, 0]

    ti = im["timestamp"].to_numpy()
    xy = np.stack([np.interp(ti, ts, P[:, k]) for k in range(2)], 1)
    yaw_i = np.angle(np.interp(ti, ts, np.cos(yaw)) + 1j * np.interp(ti, ts, np.sin(yaw)))
    return dict(
        name=ts_folder[-9:],
        ts_folder=ts_folder,
        split=split,
        t=ti,
        paths=(str(run_dir) + "/" + im["image"]).to_numpy(),
        xy=xy,
        yaw=yaw_i,
    )


class Frames(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = cv2.cvtColor(cv2.imread(self.paths[i]), cv2.COLOR_BGR2RGB)
        return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0


def describe(runs: list[dict], model_name: str, batch: int, device: str) -> None:
    """Attach an L2-normalized pooled DINO descriptor per frame."""
    model = load_dino(model_name, device=device)
    patch = patch_size(model)
    for run in runs:
        loader = DataLoader(Frames(run["paths"]), batch_size=batch, num_workers=8)
        feats = [dino_pooled_features(model, x.to(device), patch).float().cpu() for x in loader]
        f = torch.cat(feats)
        run["desc"] = torch.nn.functional.normalize(f, dim=1).numpy()
        print(f"  {run['name']}: {run['desc'].shape[0]} descriptors")


def mutual_matches(a: dict, b: dict, min_sim: float, topk: int = 1):
    """Mutual nearest neighbours between two runs' descriptors."""
    S = a["desc"] @ b["desc"].T
    ij = S.argmax(1)
    ji = S.argmax(0)
    i = np.arange(len(ij))
    keep = (ji[ij] == i) & (S[i, ij] >= min_sim)
    return i[keep], ij[keep], S[i[keep], ij[keep]]


def _rot(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s], [s, c]])


def ransac_se2(pa, ya, pb, yb, dist_thresh: float, ang_thresh: float):
    """SE(2) taking run A into run B, by 1-point RANSAC over place matches."""
    best = (0, None)
    for k in range(len(pa)):
        th = yb[k] - ya[k]
        R = _rot(th)
        t = pb[k] - R @ pa[k]
        d = np.linalg.norm(pa @ R.T + t - pb, axis=1)
        da = np.abs(np.angle(np.exp(1j * (ya + th - yb))))
        inl = (d < dist_thresh) & (da < ang_thresh)
        if inl.sum() > best[0]:
            best = (int(inl.sum()), inl)
    n, inl = best
    if inl is None or n < 3:
        return None, 0, inl
    # Kabsch refit on inliers
    A, B = pa[inl], pb[inl]
    ca, cb = A.mean(0), B.mean(0)
    H = (A - ca).T @ (B - cb)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, d]) @ U.T
    return (R, cb - R @ ca), n, inl


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cam", default="zed2")
    p.add_argument("--model", default="facebook/dinov3-vitb16-pretrain-lvd1689m")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--min-sim", type=float, default=0.6, help="Cosine floor for a place match.")
    p.add_argument("--dist-thresh", type=float, default=8.0, help="RANSAC inlier radius (m).")
    p.add_argument("--ang-thresh", type=float, default=np.deg2rad(30), help="RANSAC inlier heading tol (rad).")
    p.add_argument("--cache", type=pathlib.Path, default=DATALOCAL_ROOT / "wayfaster_split_videos" / "desc.npz")
    p.add_argument("--out", type=pathlib.Path, default=DATALOCAL_ROOT / "wayfaster_split_videos" / "align.npz")
    args = p.parse_args()

    runs = [load_run(args.cam, s, f) for s, f in RUNS_2023_06_28]

    if args.cache.exists():
        z = np.load(args.cache)
        for r in runs:
            r["desc"] = z[r["name"]]
        print(f"loaded descriptors from {args.cache}")
    else:
        print("extracting DINO descriptors")
        describe(runs, args.model, args.batch, "cuda")
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.cache, **{r["name"]: r["desc"] for r in runs})

    n = len(runs)
    print(f"\n{'pair':<24}{'matches':>9}{'inliers':>9}{'frac':>7}   transform")
    results = {}
    for i in range(n):
        for j in range(i + 1, n):
            a, b = runs[i], runs[j]
            ia, ib, sim = mutual_matches(a, b, args.min_sim)
            if len(ia) < 3:
                print(f"{a['name']}->{b['name']:<14}{len(ia):>9}{'-':>9}{'-':>7}")
                continue
            T, ninl, inl = ransac_se2(
                a["xy"][ia], a["yaw"][ia], b["xy"][ib], b["yaw"][ib], args.dist_thresh, args.ang_thresh
            )
            frac = ninl / max(len(ia), 1)
            desc = "-"
            if T is not None:
                R, t = T
                desc = f"rot={np.degrees(np.arctan2(R[1,0],R[0,0])):7.1f}deg t=[{t[0]:7.1f},{t[1]:7.1f}]"
                results[f"{a['name']}->{b['name']}"] = np.concatenate([R.ravel(), t])
            print(f"{a['name']}->{b['name']:<14}{len(ia):>9}{ninl:>9}{frac:>7.2f}   {desc}")

    if results:
        np.savez(args.out, **results)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
