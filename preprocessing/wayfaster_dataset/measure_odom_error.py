#!/usr/bin/env python
"""Measure odometry error as the gap to the SIFT+depth (PnP) relative pose on near-coincident frame pairs."""

import argparse
import itertools
import pathlib
import sys

import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from align_runs import RUNS_2023_06_28, load_run
from plot_bev_map import run_poses

SIFT_UPSCALE = 3


def features(path: str, sift):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    big = cv2.resize(img, None, fx=SIFT_UPSCALE, fy=SIFT_UPSCALE, interpolation=cv2.INTER_CUBIC)
    kp, des = sift.detectAndCompute(big, None)
    pts = np.array([k.pt for k in kp], np.float32) / SIFT_UPSCALE if kp else np.zeros((0, 2), np.float32)
    return pts, des


def relative_pose(a, b, sift, K, C, min_inliers: int):
    """T_camB_from_camA from matched features and A's depth."""
    pa, da = features(a["img"], sift)
    pb, db = features(b["img"], sift)
    if da is None or db is None or len(pa) < 8 or len(pb) < 8:
        return None

    m = cv2.BFMatcher().knnMatch(da, db, k=2)
    good = [x for x, y in (p for p in m if len(p) == 2) if x.distance < 0.8 * y.distance]
    if len(good) < min_inliers:
        return None

    depth = cv2.imread(a["depth"], cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
    obj, img = [], []
    for g in good:
        u, v = pa[g.queryIdx]
        ui, vi = int(round(u)), int(round(v))
        if not (0 <= ui < depth.shape[1] and 0 <= vi < depth.shape[0]):
            continue
        z = depth[vi, ui]
        if not (0.3 < z < 20.0):
            continue
        obj.append([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z])
        img.append(pb[g.trainIdx])
    if len(obj) < min_inliers:
        return None

    ok, rvec, tvec, inl = cv2.solvePnPRansac(
        np.array(obj, np.float32), np.array(img, np.float32), K, None,
        reprojectionError=2.0, iterationsCount=2000, flags=cv2.SOLVEPNP_EPNP,
    )
    if not ok or inl is None or len(inl) < min_inliers:
        return None
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(rvec)[0]
    T[:3, 3] = tvec.ravel()
    return T, len(inl), len(obj)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cam", default="zed2")
    p.add_argument("--max-dist", type=float, default=1.0)
    p.add_argument("--max-ang", type=float, default=np.deg2rad(15))
    p.add_argument("--min-inliers", type=int, default=15)
    p.add_argument("--n-pairs", type=int, default=120)
    p.add_argument("--mode", choices=["inter", "intra"], default="inter")
    args = p.parse_args()

    runs = [load_run(args.cam, s, f) for s, f in RUNS_2023_06_28]
    meta = {}
    for (s, f), r in zip(RUNS_2023_06_28, runs):
        paths, dpaths, xyz, Rw, K, C = run_poses(args.cam, s, f)
        meta[r["name"]] = dict(paths=paths, depth=dpaths, xyz=xyz, R=Rw, K=K, C=C)
    K = meta[runs[0]["name"]]["K"]
    C = meta[runs[0]["name"]]["C"]

    pairs = []
    combos = itertools.combinations(range(9), 2) if args.mode == "inter" else [(i, i) for i in range(9)]
    for ia, ib in combos:
        a, b = runs[ia], runs[ib]
        for i, js in enumerate(cKDTree(a["xy"]).query_ball_tree(cKDTree(b["xy"]), args.max_dist)):
            for j in js:
                if ia == ib and abs(a["t"][i] - b["t"][j]) < 30:
                    continue
                if abs(np.angle(np.exp(1j * (a["yaw"][i] - b["yaw"][j])))) < args.max_ang:
                    pairs.append((a["name"], i, b["name"], j))
    rng = np.random.default_rng(0)
    rng.shuffle(pairs)
    pairs = pairs[: args.n_pairs]
    print(f"{len(pairs)} candidate pairs ({args.mode}, dist<{args.max_dist} m, ang<{np.degrees(args.max_ang):.0f} deg)")

    sift = cv2.SIFT_create(nfeatures=3000)
    dt_err, dr_err, used = [], [], 0
    for na, i, nb, j in pairs:
        A, B = meta[na], meta[nb]
        res = relative_pose(
            dict(img=A["paths"][i], depth=A["depth"][i]), dict(img=B["paths"][j]), sift, K, C, args.min_inliers
        )
        if res is None:
            continue
        T_vis, ninl, ntot = res

        TA = np.eye(4)
        TA[:3, :3] = A["R"][i].as_matrix()
        TA[:3, 3] = A["xyz"][i]
        TB = np.eye(4)
        TB[:3, :3] = B["R"][j].as_matrix()
        TB[:3, 3] = B["xyz"][j]
        T_odom = np.linalg.inv(TB @ C) @ (TA @ C)  # camA -> camB per odometry

        D = np.linalg.inv(T_vis) @ T_odom
        dt_err.append(np.linalg.norm(D[:3, 3]))
        dr_err.append(np.degrees(np.linalg.norm(Rot.from_matrix(D[:3, :3]).as_rotvec())))
        used += 1

    if not used:
        print("no pairs solved")
        return
    dt_err, dr_err = np.array(dt_err), np.array(dr_err)
    print(f"\nsolved {used}/{len(pairs)} pairs")
    print("odometry vs vision disagreement:")
    for q in [25, 50, 75, 90]:
        print(f"  p{q}: translation {np.percentile(dt_err, q):5.2f} m   rotation {np.percentile(dr_err, q):5.1f} deg")


if __name__ == "__main__":
    main()
