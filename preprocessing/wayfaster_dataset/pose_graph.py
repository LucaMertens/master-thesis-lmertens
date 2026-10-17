#!/usr/bin/env python
"""Joint SE(2) pose-graph optimization of all runs from odometry and LoFTR/PnP loop closures."""

import argparse
import itertools
import pathlib
import sys

import gtsam
import numpy as np
from shared_utils.paths import DATALOCAL_ROOT
from scipy.spatial import cKDTree

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from align_runs import RUNS_2023_06_28, load_run
from matcher import Matcher, relative_pose
from plot_bev_map import run_poses


def se2(T):
    return np.array([T[0, 3], T[1, 3], np.arctan2(T[1, 0], T[0, 0])])


def pose2_mat(x, y, th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s, 0, x], [s, c, 0, y], [0, 0, 1, 0], [0, 0, 0, 1]])


def candidates(runs, args):
    """Loop-closure candidates from current odometry geometry."""
    out = []
    for ia, ib in itertools.combinations(range(len(runs)), 2):
        a, b = runs[ia], runs[ib]
        hits = []
        for i, js in enumerate(cKDTree(a["xy"]).query_ball_tree(cKDTree(b["xy"]), args.radius)):
            for j in js:
                if abs(np.angle(np.exp(1j * (a["yaw"][i] - b["yaw"][j])))) < args.max_ang:
                    hits.append((ia, i, ib, j))
        out += hits[:: max(1, len(hits) // args.per_pair)][: args.per_pair]
    for ia, a in enumerate(runs):
        hits = []
        for i, j in cKDTree(a["xy"]).query_pairs(args.radius):
            if abs(a["t"][i] - a["t"][j]) < args.min_dt:
                continue
            if abs(np.angle(np.exp(1j * (a["yaw"][i] - a["yaw"][j])))) < args.max_ang:
                hits.append((ia, i, ia, j))
        out += hits[:: max(1, len(hits) // args.per_pair)][: args.per_pair]
    return out


def build_closures(runs, meta, K, C, args):
    cache = args.closures
    if cache.exists():
        z = np.load(cache)
        print(f"loaded {len(z['idx'])} closures from {cache}")
        return z["idx"], z["rel"]

    cands = candidates(runs, args)
    print(f"{len(cands)} candidate closures, matching with LoFTR")
    m = Matcher()
    idx, rel = [], []
    for n, (ia, i, ib, j) in enumerate(cands):
        A, B = meta[runs[ia]["name"]], meta[runs[ib]["name"]]
        res = relative_pose(m, A["paths"][i], A["depth"][i], B["paths"][j], K, min_inliers=args.min_inliers)
        if res is None:
            continue
        T_vis = res[0]
        T_BA = C @ T_vis @ np.linalg.inv(C)  # baseA -> baseB
        meas = se2(np.linalg.inv(T_BA))  # pose_a.between(pose_b)

        # gate against current odometry
        pa = pose2_mat(*runs[ia]["xy"][i], runs[ia]["yaw"][i])
        pb = pose2_mat(*runs[ib]["xy"][j], runs[ib]["yaw"][j])
        odo = se2(np.linalg.inv(pa) @ pb)
        if np.linalg.norm(meas[:2] - odo[:2]) > args.gate_t:
            continue
        if abs(np.angle(np.exp(1j * (meas[2] - odo[2])))) > args.gate_r:
            continue

        idx.append((ia, i, ib, j))
        rel.append(meas)
        if (n + 1) % 500 == 0:
            print(f"  {n+1}/{len(cands)} candidates, {len(idx)} accepted")

    idx, rel = np.array(idx), np.array(rel)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, idx=idx, rel=rel)
    print(f"accepted {len(idx)}/{len(cands)} closures -> {cache}")
    return idx, rel


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cam", default="zed2")
    p.add_argument("--radius", type=float, default=3.0)
    p.add_argument("--max-ang", type=float, default=np.deg2rad(35))
    p.add_argument("--min-dt", type=float, default=20.0)
    p.add_argument("--per-pair", type=int, default=150, help="Max closures per run or run-pair.")
    p.add_argument("--min-inliers", type=int, default=25)
    p.add_argument("--gate-t", type=float, default=4.0, help="Reject closure if it moves a pose more than this (m).")
    p.add_argument("--gate-r", type=float, default=np.deg2rad(40))
    p.add_argument("--odom-sigma", type=float, nargs=3, default=[0.02, 0.02, 0.005])
    p.add_argument("--loop-sigma", type=float, nargs=3, default=[0.20, 0.20, 0.05])
    p.add_argument("--huber", type=float, default=1.0)
    p.add_argument("--closures", type=pathlib.Path, default=DATALOCAL_ROOT / "wayfaster_split_videos" / "closures.npz")
    p.add_argument("--out", type=pathlib.Path, default=DATALOCAL_ROOT / "wayfaster_split_videos" / "poses_opt.npz")
    args = p.parse_args()

    runs = [load_run(args.cam, s, f) for s, f in RUNS_2023_06_28]
    meta = {}
    for (s, f), r in zip(RUNS_2023_06_28, runs):
        paths, dpaths, xyz, Rw, K, C = run_poses(args.cam, s, f)
        meta[r["name"]] = dict(paths=paths, depth=dpaths)
    K, C = run_poses(args.cam, *RUNS_2023_06_28[0])[4:6]

    idx, rel = build_closures(runs, meta, K, C, args)

    offset, k = {}, 0
    for i, r in enumerate(runs):
        offset[i] = k
        k += len(r["xy"])
    print(f"graph: {k} poses, {len(idx)} closures")

    graph = gtsam.NonlinearFactorGraph()
    values = gtsam.Values()
    odom_n = gtsam.noiseModel.Diagonal.Sigmas(np.array(args.odom_sigma))
    loop_n = gtsam.noiseModel.Robust.Create(
        gtsam.noiseModel.mEstimator.Huber.Create(args.huber),
        gtsam.noiseModel.Diagonal.Sigmas(np.array(args.loop_sigma)),
    )
    anchor_n = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-3, 1e-3, 1e-4]))
    weak_n = gtsam.noiseModel.Diagonal.Sigmas(np.array([5.0, 5.0, 0.3]))

    for ri, r in enumerate(runs):
        for i in range(len(r["xy"])):
            values.insert(offset[ri] + i, gtsam.Pose2(r["xy"][i, 0], r["xy"][i, 1], r["yaw"][i]))
        # sequential odometry
        for i in range(len(r["xy"]) - 1):
            a = pose2_mat(*r["xy"][i], r["yaw"][i])
            b = pose2_mat(*r["xy"][i + 1], r["yaw"][i + 1])
            d = se2(np.linalg.inv(a) @ b)
            graph.add(gtsam.BetweenFactorPose2(offset[ri] + i, offset[ri] + i + 1, gtsam.Pose2(*d), odom_n))
        # anchor each run
        n = anchor_n if ri == 0 else weak_n
        graph.add(gtsam.PriorFactorPose2(offset[ri], gtsam.Pose2(r["xy"][0, 0], r["xy"][0, 1], r["yaw"][0]), n))

    for (ia, i, ib, j), d in zip(idx, rel):
        graph.add(gtsam.BetweenFactorPose2(offset[ia] + i, offset[ib] + j, gtsam.Pose2(*d), loop_n))

    params = gtsam.LevenbergMarquardtParams()
    params.setVerbosityLM("SUMMARY")
    params.setMaxIterations(100)
    opt = gtsam.LevenbergMarquardtOptimizer(graph, values, params)
    print(f"initial error {graph.error(values):.1f}")
    result = opt.optimize()
    print(f"final error   {graph.error(result):.1f}")

    out = {}
    for ri, r in enumerate(runs):
        P = np.array([[result.atPose2(offset[ri] + i).x(), result.atPose2(offset[ri] + i).y(),
                       result.atPose2(offset[ri] + i).theta()] for i in range(len(r["xy"]))])
        out[r["name"]] = P
        shift = np.linalg.norm(P[:, :2] - r["xy"], axis=1)
        print(f"  {r['name']}: median shift {np.median(shift):.2f} m, max {shift.max():.2f} m")
    np.savez(args.out, **out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
