"""Project region world points to pixels in the annotated frame (geometric only, cached).

Run: uv run python side_experiments/wrizz_image/region_uv.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "wrizz_labeling"))
import common
from traversability_network.adaptations import propagate as P

INDEX = "side_experiments/wrizz_labeling/output/index_k64"
OUT = "side_experiments/wrizz_image/cache/region_uv.npz"
NATIVE = (1080, 1440)
BORDER = 4


def project(p_world, frame, geom):
    """(N, 3) world -> (N, 2) pixels and a validity mask for one frame."""
    homo = np.concatenate([p_world, np.ones((len(p_world), 1))], 1)
    cam = homo @ geom.T_cam_world[frame].T
    z = cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = geom.K[0, 0] * cam[:, 0] / z + geom.K[0, 2]
        v = geom.K[1, 1] * cam[:, 1] / z + geom.K[1, 2]
    ok = (z > 0) & (u >= BORDER) & (u < NATIVE[1] - BORDER)
    ok &= (v >= BORDER) & (v < NATIVE[0] - BORDER)
    return np.stack([u, v], 1), ok & np.isfinite(u) & np.isfinite(v)


def main():
    cfg = common.load_cfg(common.CFG_FILE)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out = {}
    for name in sorted(os.listdir(INDEX)):
        if not name.endswith(".npz"):
            continue
        mission = name[:-4]
        d = np.load(os.path.join(INDEX, name), allow_pickle=True)
        geom = P.MissionGeometry(os.path.join(cfg.DATASET.DATA_ROOT, mission),
                                 cfg.DATASET.DEPTH_CACHE_ROOT, cache=1)
        k = d["region_pts"].shape[1]
        uv = np.zeros((len(d["label_id"]), k + 1, 2), np.float32)
        ok = np.zeros((len(d["label_id"]), k + 1), bool)
        for i, ts in enumerate(d["anchor_ts_ns"]):
            f = geom.frame_index(int(ts))
            n = int(d["n_region"][i])
            pts = np.concatenate([d["p_world"][i][None], d["region_pts"][i][:n]])
            p, valid = project(pts, f, geom)
            uv[i, : len(p)], ok[i, : len(p)] = p, valid
        # the click overrides the reprojection
        uv[:, 0], ok[:, 0] = d["anchor_uv"], True
        out[f"{mission}/uv"], out[f"{mission}/ok"] = uv, ok
        out[f"{mission}/label_id"] = d["label_id"]
        out[f"{mission}/side"] = d["side"]
        print(f"{mission}: {len(uv)} endpoints, {ok[:, 1:].sum() / max(len(uv), 1):.1f} "
              f"region pixels each")
    np.savez_compressed(OUT, **out)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
