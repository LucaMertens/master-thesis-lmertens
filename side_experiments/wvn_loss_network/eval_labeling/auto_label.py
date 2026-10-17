"""Seed eval labels from depth geometry for manual refinement."""

import argparse
import json
import os

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

DMIN, DMAX = 0.3, 20.0  # m, valid depth range
NORMAL_SUPPORT = 15     # px blur on the normal field
BORDER = 8              # px, rectification edge
UNTRAV = 4


def up_in_base(mission_dir, ts_ns):
    """Gravity up in the base frame at ts (odom ts are seconds)."""
    o = np.load(os.path.join(mission_dir, "dlio_odom.npy"))
    j = int(np.argmin(np.abs(o[:, 0] - ts_ns / 1e9)))
    return Rotation.from_quat(o[j, 4:8]).as_matrix()[2, :]


def obstacle_map(mission_dir, ts_ns, slope_deg, max_dist, min_area, min_height):
    dm = np.load(os.path.join(mission_dir, "depth_manifest.npz"))["depth"]
    j = int(np.argmin(np.abs(dm["ts_ns"].astype(np.int64) - ts_ns)))
    d = cv2.imread(os.path.join(mission_dir, dm["path"][j]), -1)
    K = np.load(os.path.join(mission_dir, "intrinsics_front_center.npy"))
    c2b = np.load(os.path.join(mission_dir, "cam2base.npy"))
    up = up_in_base(mission_dir, ts_ns)

    Z = d.astype(np.float32) / 1000.0
    valid = (Z >= DMIN) & (Z <= DMAX) & (Z < max_dist)
    valid[:BORDER] = valid[-BORDER:] = valid[:, :BORDER] = valid[:, -BORDER:] = False
    # full-support normals only, else the valid/border edge reads as a wall
    support = cv2.erode(valid.astype(np.uint8), np.ones((NORMAL_SUPPORT, NORMAL_SUPPORT), np.uint8)).astype(bool)
    h, w = Z.shape
    us, vs = np.meshgrid(np.arange(w), np.arange(h))
    pc = np.stack([(us - K[0, 2]) * Z / K[0, 0], (vs - K[1, 2]) * Z / K[1, 1], Z], -1)
    pb = pc @ c2b[:3, :3].T + c2b[:3, 3]

    # terrain normal from depth gradients, tilt from gravity = slope
    n = np.cross(np.gradient(pb, axis=1), np.gradient(pb, axis=0))
    n = cv2.blur(n, (NORMAL_SUPPORT, NORMAL_SUPPORT))
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-8
    tilt = np.degrees(np.arccos(np.clip(np.abs(n @ up), 0, 1)))
    gh = pb @ up

    steep = (support & (tilt > slope_deg)).astype(np.uint8)
    steep = cv2.morphologyEx(steep, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    nc, lab, stats, _ = cv2.connectedComponentsWithStats(steep, connectivity=8)
    # height span per steep component
    lf, gf = lab.ravel(), gh.ravel()
    hi = np.full(nc, -np.inf)
    np.maximum.at(hi, lf, gf)
    lo = np.full(nc, np.inf)
    np.minimum.at(lo, lf, gf)
    tall = (stats[:, cv2.CC_STAT_AREA] >= min_area) & (hi - lo >= min_height)
    tall[0] = False
    return tall[lab]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="side_experiments/wvn_loss_network/wvn_dino.yaml")
    ap.add_argument("--out", default="side_experiments/wvn_loss_network/output/eval_labeling")
    ap.add_argument("--slope-deg", type=float, default=55.0, help="surface tilt from gravity = untraversable")
    ap.add_argument("--max-dist", type=float, default=6.0, help="m, ignore noisier far depth")
    ap.add_argument("--min-area", type=int, default=2000, help="px (native res), drop smaller blobs as speckle")
    ap.add_argument("--min-height", type=float, default=0.6, help="m, steep face span below this = step-over-able")
    ap.add_argument("--force", action="store_true", help="also seed frames that already have labels")
    args = ap.parse_args()

    data_root = yaml.safe_load(open(args.cfg))["DATASET"]["DATA_ROOT"]
    frames = json.load(open(os.path.join(args.out, "frames.json")))
    labels_dir = os.path.join(args.out, "labels")
    os.makedirs(labels_dir, exist_ok=True)

    for e in frames:
        dst = os.path.join(labels_dir, e["name"] + ".png")
        if os.path.exists(dst) and not args.force:
            continue
        m = obstacle_map(os.path.join(data_root, e["mission"]), e["ts_ns"],
                         args.slope_deg, args.max_dist, args.min_area, args.min_height)
        seed = cv2.resize(np.where(m, UNTRAV, 0).astype(np.uint8), (e["w"], e["h"]), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(dst, seed)
        print(f"{e['name']}: {int((seed > 0).mean() * 100)}% untraversable", flush=True)
    print("done")


if __name__ == "__main__":
    main()
