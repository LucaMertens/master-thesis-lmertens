"""Ceiling probe, step 1: cache frozen DINO patch features for low/high-mu path points.

Run: uv run python side_experiments/dino_mu_probe/extract_features.py --cfg <cfg.yaml>"""

import argparse
import os
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from scipy.spatial.transform import Rotation, Slerp
from tqdm import tqdm

NATIVE = (1440, 1080)  # Alphasense native resolution the intrinsics refer to
BASE_HEIGHT = 0.5      # m, base origin above ground contact
LOOKBACK = (0.3, 15.0)  # s before arrival to search for a view (slip events are slow/turning)
VIEW_DEPTH = (1.0, 6.0)  # m, accepted camera-frame depth of the target point
EDGE_MARGIN = 8        # px, keep projections away from image borders
HIGH_MU = 0.85


def interp_odom(odom, query_t):
    """dlio pose (M,8: t,x,y,z,qx,qy,qz,qw) -> (position (N,3), quaternion (N,4) xyzw)."""
    t = odom[:, 0]
    order = np.argsort(t)
    t, pos, quat = t[order], odom[order, 1:4], odom[order, 4:8]
    t, uniq = np.unique(t, return_index=True)
    pos, quat = pos[uniq], quat[uniq]
    clamped = np.clip(query_t, t[0], t[-1])
    position = np.column_stack([np.interp(clamped, t, pos[:, i]) for i in range(3)])
    quaternion = Slerp(t, Rotation.from_quat(quat))(clamped).as_quat()
    return position, quaternion


def select_points(mu, obs, mu_max, obs_min, neg_per_pos, rng, mid_band=False):
    low = np.flatnonzero((mu <= mu_max) & (obs >= obs_min))
    high = np.flatnonzero((mu >= HIGH_MU) & (obs >= obs_min))
    n_neg = min(len(high), max(200, neg_per_pos * len(low)))
    if len(high) > n_neg:
        high = np.sort(rng.choice(high, n_neg, replace=False))
    mid = np.flatnonzero((mu > mu_max) & (mu < HIGH_MU) & (obs >= obs_min)) if mid_band else np.array([], np.int64)
    if len(mid) > n_neg:
        mid = np.sort(rng.choice(mid, n_neg, replace=False))
    return low, mid, high


def find_view(ground, t_arrive, img_ts, R_wc, t_wc, K, img_size):
    """Finds the latest frame with a clean view of a ground point."""
    W, H = img_size
    lo = np.searchsorted(img_ts, t_arrive - LOOKBACK[1])
    hi = np.searchsorted(img_ts, t_arrive - LOOKBACK[0])
    if hi <= lo:
        return None
    p = np.einsum("fji,fj->fi", R_wc[lo:hi], ground - t_wc[lo:hi])
    with np.errstate(divide="ignore", invalid="ignore"):
        u = K[0, 0] * p[:, 0] / p[:, 2] + K[0, 2]
        v = K[1, 1] * p[:, 1] / p[:, 2] + K[1, 2]
    ok = (
        (p[:, 2] >= VIEW_DEPTH[0]) & (p[:, 2] <= VIEW_DEPTH[1])
        & (u >= EDGE_MARGIN) & (u < W - EDGE_MARGIN)
        & (v >= EDGE_MARGIN) & (v < H - EDGE_MARGIN)
    )
    if not ok.any():
        return None
    f = np.flatnonzero(ok)[-1]
    return lo + f, float(u[f]), float(v[f])


def mission_records(cfg, mission, rng, neg_per_pos, mid_band=False):
    """(img_path, u, v, mu, y, mission) tuples for all projectable probe points."""
    ds = cfg["DATASET"]
    data_dir = os.path.join(ds["DATA_ROOT"], mission)
    rhe_dir = os.path.join(ds["LABELS_ROOT"], mission, "rhe")

    ts = np.load(os.path.join(rhe_dir, "rhe_timestamp.npy")).astype(np.float64)
    order = np.argsort(ts)
    ts = ts[order]
    mu = np.load(os.path.join(rhe_dir, "rhe_trav_labels.npy"))[order, 0]
    obs = np.load(os.path.join(rhe_dir, "rhe_mu_observability.npy"))[order]
    odom = np.load(os.path.join(data_dir, "dlio_odom.npy"))
    cam2base = np.load(os.path.join(data_dir, "cam2base.npy"))
    K = np.load(os.path.join(data_dir, "intrinsics.npy"))

    fc = np.load(os.path.join(data_dir, "image_manifest.npz"))["front_center"]
    img_ts = fc["ts_ns"].astype(np.float64) / 1e9
    img_order = np.argsort(img_ts)
    img_ts = img_ts[img_order]
    root = os.path.join(ds["RESIZED_ROOT"], mission)
    img_paths = [os.path.join(root, p) for p in fc["path"][img_order]]

    # camera pose in the world per image frame
    ipos, iquat = interp_odom(odom, img_ts)
    R_wb = Rotation.from_quat(iquat).as_matrix()
    R_wc = R_wb @ cam2base[:3, :3]
    t_wc = R_wb @ cam2base[:3, 3] + ipos

    W, H = cfg["MODEL"]["INPUT_SIZE"]
    Kr = K.copy()
    Kr[0] *= W / NATIVE[0]
    Kr[1] *= H / NATIVE[1]

    pos, _ = interp_odom(odom, ts)
    low, mid, high = select_points(
        mu, obs, ds.get("OVERSAMPLE_MU_MAX", 0.6), ds.get("OVERSAMPLE_OBSERVABILITY_THRESH", 0.6),
        neg_per_pos, rng, mid_band=mid_band,
    )
    recs, placed = [], {1: 0, 0: 0, -1: 0}
    for j, y in [(j, 1) for j in low] + [(j, 0) for j in high] + [(j, -1) for j in mid]:
        ground = pos[j] - np.array([0.0, 0.0, BASE_HEIGHT])
        view = find_view(ground, ts[j], img_ts, R_wc, t_wc, Kr, (W, H))
        if view is not None:
            f, u, v = view
            recs.append((img_paths[f], u, v, float(mu[j]), y, mission))
            placed[y] += 1
    print(f"  {mission}: low {placed[1]}/{len(low)} placed, {placed[0]}/{len(high)} high, "
          f"{placed[-1]}/{len(mid)} mid, {len(recs)} total")
    return recs


class ResNetFeatures(torch.nn.Module):
    """Torchvision ResNet wrapped with the DINO models' feature-extraction API."""

    patch_size = 32

    def __init__(self, resnet):
        super().__init__()
        self.stem = torch.nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = (
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
        )

    def get_intermediate_layers(self, x, n=1, reshape=True):
        x = self.layer4(self.layer3(self.layer2(self.layer1(self.stem(x)))))
        return [x]


def load_model(name, weights, device):
    if name.startswith("resnet"):
        import torchvision.models as tvm

        model = getattr(tvm, name)(weights=weights or "IMAGENET1K_V1")
        return ResNetFeatures(model).eval().to(device)
    from ood_detection.common.dino import load_dino

    return load_dino(name, weights, device)


@torch.no_grad()
def extract_features(recs, model, patch, device, batch=32):
    """Bilinear-sample the last-layer patch feature at each record's pixel."""
    from ood_detection.common.dino import dino_patch_features

    by_frame = defaultdict(list)
    for i, r in enumerate(recs):
        by_frame[r[0]].append(i)
    paths = list(by_frame)
    feats = None
    for k in tqdm(range(0, len(paths), batch), desc="frames"):
        imgs, groups = [], []
        for p in paths[k : k + batch]:
            img = cv2.imread(p)
            assert img is not None, p
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
            H, W = img.shape[:2]
            img = cv2.resize(img, (W // patch * patch, H // patch * patch), interpolation=cv2.INTER_AREA)
            imgs.append(img)
            idx = by_frame[p]
            # pixel -> [-1,1] over the full image = full feature map (align_corners=False)
            uv = np.array([[recs[i][1] / W * 2 - 1, recs[i][2] / H * 2 - 1] for i in idx], np.float32)
            groups.append((idx, uv))
        x = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).to(device)
        fmap = dino_patch_features(model, x, patch)
        if feats is None:
            feats = np.zeros((len(recs), fmap.shape[1]), np.float32)
        for b, (idx, uv) in enumerate(groups):
            grid = torch.from_numpy(uv).to(device).view(1, 1, -1, 2)
            f = F.grid_sample(fmap[b : b + 1], grid, align_corners=False)[0, :, 0].T
            feats[idx] = f.cpu().numpy()
    return feats


def save_overlays(recs, out_dir, n, rng):
    """A few frames with projected probe points drawn on, for geometry sanity checks."""
    by_frame = defaultdict(list)
    for r in recs:
        by_frame[r[0]].append(r)
    low_frames = [p for p, rs in by_frame.items() if any(r[4] for r in rs)]
    os.makedirs(out_dir, exist_ok=True)
    for p in rng.choice(low_frames, min(n, len(low_frames)), replace=False):
        img = cv2.imread(p)
        for _, u, v, mu_val, y, _ in by_frame[p]:
            color = (0, 0, 255) if y else (0, 255, 0)
            cv2.circle(img, (round(u), round(v)), 5, color, 2)
            cv2.putText(img, f"{mu_val:.2f}", (round(u) + 6, round(v) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(out_dir, os.path.basename(p)), img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="training/traversability_network/configs/grandtour_lite.yaml")
    ap.add_argument("--model", default="dinov2_vitb14_reg", help="dinov2_*/dinov3_* or resnet18/34/50")
    ap.add_argument("--weights", default=None,
                     help="local .pth for gated dinov3_* models, or a torchvision weights "
                          "enum name for resnet* (default IMAGENET1K_V1)")
    ap.add_argument("--missions", nargs="*", default=None, help="default: TRAIN + VALID missions")
    ap.add_argument("--neg-per-pos", type=int, default=4)
    ap.add_argument("--mid-band", action="store_true",
                    help="also sample mu in (0.6, 0.85) for regression probes")
    ap.add_argument("--out", default="side_experiments/dino_mu_probe/output/features.npz")
    ap.add_argument("--overlay-dir", default="side_experiments/dino_mu_probe/output/overlays")
    ap.add_argument("--overlays", type=int, default=8)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.cfg))
    missions = args.missions or list(
        dict.fromkeys(cfg["DATASET"]["TRAIN_MISSIONS"] + cfg["DATASET"]["VALID_MISSIONS"])
    )
    rng = np.random.default_rng(args.seed)

    recs = []
    for mission in missions:
        recs += mission_records(cfg, mission, rng, args.neg_per_pos, mid_band=args.mid_band)
    print(f"{len(recs)} probe points on {len({r[0] for r in recs})} frames")
    if args.overlays:
        save_overlays(recs, args.overlay_dir, args.overlays, rng)

    from ood_detection.common.dino import patch_size

    device = torch.device(args.device)
    model = load_model(args.model, args.weights, device)
    patch = patch_size(model)
    feats = extract_features(recs, model, patch, device)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(
        args.out,
        features=feats,
        mu=np.array([r[3] for r in recs], np.float32),
        y=np.array([r[4] for r in recs], np.int64),
        mission=np.array([r[5] for r in recs]),
        u=np.array([r[1] for r in recs], np.float32),
        v=np.array([r[2] for r in recs], np.float32),
        path=np.array([r[0] for r in recs]),
        model=args.model,
    )
    print("saved ->", args.out)


if __name__ == "__main__":
    main()
