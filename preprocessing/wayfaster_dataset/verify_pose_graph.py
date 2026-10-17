#!/usr/bin/env python
"""Compare odometry before and after pose-graph optimization on pairs not used as loop closures."""

import argparse
import itertools
import pathlib
import sys

import numpy as np
from shared_utils.paths import DATALOCAL_ROOT
from scipy.spatial import cKDTree

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from align_runs import RUNS_2023_06_28, load_run
from matcher import Matcher, relative_pose
from plot_bev_map import run_poses
from pose_graph import pose2_mat, se2


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cam", default="zed2")
    p.add_argument("--n-pairs", type=int, default=150)
    p.add_argument("--radius", type=float, default=1.0)
    p.add_argument("--max-ang", type=float, default=np.deg2rad(15))
    p.add_argument("--closures", type=pathlib.Path, default=DATALOCAL_ROOT / "wayfaster_split_videos" / "closures.npz")
    p.add_argument("--poses", type=pathlib.Path, default=DATALOCAL_ROOT / "wayfaster_split_videos" / "poses_opt.npz")
    args = p.parse_args()

    runs = [load_run(args.cam, s, f) for s, f in RUNS_2023_06_28]
    opt = np.load(args.poses)
    meta = {}
    for (s, f), r in zip(RUNS_2023_06_28, runs):
        paths, dpaths, *_ = run_poses(args.cam, s, f)
        meta[r["name"]] = dict(paths=paths, depth=dpaths)
    K, C = run_poses(args.cam, *RUNS_2023_06_28[0])[4:6]

    used = {tuple(x) for x in np.load(args.closures)["idx"]}

    cands = []
    for ia, ib in itertools.combinations(range(9), 2):
        a, b = runs[ia], runs[ib]
        for i, js in enumerate(cKDTree(a["xy"]).query_ball_tree(cKDTree(b["xy"]), args.radius)):
            for j in js:
                if (ia, i, ib, j) in used:
                    continue
                if abs(np.angle(np.exp(1j * (a["yaw"][i] - b["yaw"][j])))) < args.max_ang:
                    cands.append((ia, i, ib, j))
    rng = np.random.default_rng(1)
    rng.shuffle(cands)
    cands = cands[: args.n_pairs]
    print(f"{len(cands)} held-out pairs (closures excluded)")

    m = Matcher()
    err = {"before": [], "after": []}
    for ia, i, ib, j in cands:
        A, B = meta[runs[ia]["name"]], meta[runs[ib]["name"]]
        res = relative_pose(m, A["paths"][i], A["depth"][i], B["paths"][j], K, min_inliers=25)
        if res is None:
            continue
        meas = se2(np.linalg.inv(C @ res[0] @ np.linalg.inv(C)))

        for tag, pa, pb in [
            ("before", pose2_mat(*runs[ia]["xy"][i], runs[ia]["yaw"][i]),
             pose2_mat(*runs[ib]["xy"][j], runs[ib]["yaw"][j])),
            ("after", pose2_mat(*opt[runs[ia]["name"]][i]), pose2_mat(*opt[runs[ib]["name"]][j])),
        ]:
            d = se2(np.linalg.inv(pa) @ pb)
            err[tag].append(
                (np.linalg.norm(d[:2] - meas[:2]), abs(np.degrees(np.angle(np.exp(1j * (d[2] - meas[2]))))))
            )

    n = len(err["before"])
    print(f"solved {n}/{len(cands)}\n")
    print(f"{'':8}{'trans p50':>11}{'trans p75':>11}{'rot p50':>10}{'rot p75':>10}")
    for tag in ["before", "after"]:
        e = np.array(err[tag])
        print(f"{tag:8}{np.median(e[:,0]):10.2f}m{np.percentile(e[:,0],75):10.2f}m"
              f"{np.median(e[:,1]):9.1f}d{np.percentile(e[:,1],75):9.1f}d")


if __name__ == "__main__":
    main()
