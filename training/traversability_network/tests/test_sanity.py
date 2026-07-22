"""Sanity checks tiers 1-3 (geometry, gradients/labels, robustness) for the GrandTour traversability training pipeline."""

import bisect
import os

import cv2
import numpy as np
import pytest
import pytorch_lightning as pl
import torch
from config.paths import EXTRACTED_ROOT, LABELS_ROOT
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from torch.utils.data import DataLoader, Subset
from traversability_network.config import get_cfg
from traversability_network.adaptations.grandtour_dataset import BORDER_CROP, GrandTourDataset
from traversability_network.network.trainer import TrainingModule

DATA_ROOT = str(EXTRACTED_ROOT)
LABELS_ROOT = str(LABELS_ROOT)
MISSION = "ETH-1"
CFG = os.path.join(os.path.dirname(__file__), "..", "configs", "grandtour.yaml")


def _cfg():
    cfg = get_cfg(CFG)
    cfg.TRAINING.WORKERS = 0
    cfg.TRAINING.PRECISION = 32
    cfg.AUGMENTATIONS.HORIZ_FLIP = 0.0
    cfg.AUGMENTATIONS.PCLOUD_DROPOUT = 0.0
    cfg.AUGMENTATIONS.MAX_TRANSLATION = 0.0
    cfg.AUGMENTATIONS.MAX_ROTATION = 0.0
    return cfg


class _Silent(TrainingModule):
    def visualize(self, *args, **kwargs):
        pass

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )


@torch.no_grad()
def _mean_trav_error(model, loader, device):
    model.to(device)
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        batch = [x.to(device) for x in batch]
        color, depth_voxel, K_full, K, ext, path, target, weights, depth_t, _ = batch
        trav_map, _, _, _ = model.model(color, depth_voxel, K_full, K, ext, depth_t)
        _, error = model.trav_criterion(path, trav_map, target, weights)
        total += float(error)
        n += 1
    return total / n


def _skip_no_data(mission=MISSION):
    if not os.path.isdir(os.path.join(DATA_ROOT, mission)):
        pytest.skip(f"{mission} not available under {DATA_ROOT}")


def _locate(ds, idx):
    bag_idx = bisect.bisect_right(ds.bag_sizes, idx) - 1
    rosbag = ds.rosbags[bag_idx]
    data_idx = idx - ds.bag_sizes[bag_idx]
    return rosbag, data_idx


# ---- Tier 1: geometry -------------------------------------------------------


@pytest.mark.tier1
def test_path_in_map_coverage():
    """Diagnostic: fraction of the 300-step horizon inside the supervised BEV map."""
    _skip_no_data()
    cfg = _cfg()
    ds = GrandTourDataset(cfg, [MISSION], train=False)

    gb = cfg.MODEL.GRID_BOUNDS
    xmin, xmax = gb["xbound"][0], gb["xbound"][1]
    ymin, ymax = gb["ybound"][0], gb["ybound"][1]
    horizon = cfg.TRAINING.HORIZON

    n_sample = min(500, len(ds))
    indices = np.linspace(0, len(ds) - 1, n_sample, dtype=int)

    fracs, finite_fracs = [], []
    for idx in indices:
        rosbag, data_idx = _locate(ds, int(idx))
        ts = [rosbag["image_timestamp"][data_idx + t] for t in range(ds.n_frames)]
        states, _, _, _ = ds.get_states(rosbag, ts, np.eye(4), False)

        finite = np.isfinite(states[:, 0])
        in_map = (
            finite
            & (states[:, 0] >= xmin)
            & (states[:, 0] <= xmax)
            & (states[:, 1] >= ymin)
            & (states[:, 1] <= ymax)
        )
        fracs.append(in_map.sum() / horizon)
        finite_fracs.append(finite.sum() / horizon)

    mean_map = float(np.mean(fracs))
    mean_fin = float(np.mean(finite_fracs))
    print(
        f"\nPath-in-map coverage (HORIZON={horizon}, {n_sample} samples sampled):\n"
        f"  finite steps : {mean_fin * 100:.1f}% mean "
        f"({np.min(finite_fracs) * 100:.0f}–{np.max(finite_fracs) * 100:.0f}%)\n"
        f"  in-map steps : {mean_map * 100:.1f}% mean "
        f"({np.min(fracs) * 100:.0f}–{np.max(fracs) * 100:.0f}%)"
    )
    # diagnostic only, no threshold assert; result informs HORIZON / xbound tuning


@pytest.mark.tier1
def test_heading_consistency():
    """RHE yaw and dlio-slerped yaw must agree within 0.1 rad (median)."""
    _skip_no_data()
    mission_dir = os.path.join(DATA_ROOT, MISSION)
    rhe_dir = os.path.join(LABELS_ROOT, MISSION, "rhe")

    odom_path = os.path.join(mission_dir, "dlio_odom.npy")
    if not os.path.exists(odom_path):
        pytest.skip(f"No ground-truth odom for {MISSION}: {odom_path} missing.")

    rhe_states = np.load(os.path.join(rhe_dir, "rhe_states.npy"))
    timestamp = np.load(os.path.join(rhe_dir, "rhe_timestamp.npy")).astype(np.float64)
    odom = np.load(odom_path)

    # Replicate GrandTourDataset orientation slerp
    t_odom = odom[:, 0]
    quat_xyzw = odom[:, 4:8]  # dlio_odom.npy stores qx,qy,qz,qw (see extract_dlio.py)
    order = np.argsort(t_odom)
    t_odom, quat_xyzw = t_odom[order], quat_xyzw[order]
    t_odom, uniq = np.unique(t_odom, return_index=True)
    quat_xyzw = quat_xyzw[uniq]
    rot = R.from_quat(quat_xyzw)
    slerp = Slerp(t_odom, rot)
    clamped = np.clip(timestamp, t_odom[0], t_odom[-1])
    yaw_odom = slerp(clamped).as_euler("zyx")[:, 0]

    yaw_rhe = rhe_states[:, 2]
    diff = np.angle(np.exp(1j * (yaw_rhe - yaw_odom)))  # wrap to [-π, π]

    median_abs = float(np.median(np.abs(diff)))
    mean_offset = float(np.angle(np.mean(np.exp(1j * diff))))
    residuals = np.angle(np.exp(1j * (diff - mean_offset)))
    residual_std = float(np.std(residuals))

    print(
        f"\nHeading consistency:\n"
        f"  median |Δyaw|  = {median_abs:.4f} rad\n"
        f"  circular mean  = {mean_offset:.4f} rad  (constant-offset indicator)\n"
        f"  residual std   = {residual_std:.4f} rad  (drift indicator)"
    )

    assert median_abs < 0.1, (
        f"Heading disagreement too large: median={median_abs:.4f} rad, "
        f"mean_offset={mean_offset:.4f} rad, residual_std={residual_std:.4f} rad"
    )


@pytest.mark.tier1
def test_position_consistency():
    """RHE x,y and dlio ground-truth x,y must agree within 0.3 m (median)."""
    _skip_no_data()
    mission_dir = os.path.join(DATA_ROOT, MISSION)
    rhe_dir = os.path.join(LABELS_ROOT, MISSION, "rhe")
    odom_path = os.path.join(mission_dir, "dlio_odom.npy")
    if not os.path.exists(odom_path):
        pytest.skip(
            f"No ground-truth odom for {MISSION}: {odom_path} missing. "
            "Rerun dlio extraction: "
            f"uv run python -m grandtour_extraction.extract_dlio "
            f"--mission-dir <bags>/{MISSION} --output-dir {mission_dir}"
        )

    rhe_states = np.load(os.path.join(rhe_dir, "rhe_states.npy"))
    timestamp = np.load(os.path.join(rhe_dir, "rhe_timestamp.npy")).astype(np.float64)
    odom = np.load(odom_path)

    order = np.argsort(timestamp)
    timestamp, rhe_states = timestamp[order], rhe_states[order]

    # Replicate GrandTourDataset._interp_odom position interpolation
    t_odom = odom[:, 0]
    o = np.argsort(t_odom)
    t_odom, pos = t_odom[o], odom[o, 1:3]
    t_odom, uniq = np.unique(t_odom, return_index=True)
    pos = pos[uniq]
    clamped = np.clip(timestamp, t_odom[0], t_odom[-1])
    x_odom = np.interp(clamped, t_odom, pos[:, 0])
    y_odom = np.interp(clamped, t_odom, pos[:, 1])

    dist = np.hypot(rhe_states[:, 0] - x_odom, rhe_states[:, 1] - y_odom)
    median, p90, dmax = (
        float(np.median(dist)),
        float(np.percentile(dist, 90)),
        float(dist.max()),
    )
    print(
        f"\nPosition consistency:\n"
        f"  median |Δpos| = {median:.4f} m\n"
        f"  p90    |Δpos| = {p90:.4f} m\n"
        f"  max    |Δpos| = {dmax:.4f} m"
    )

    assert median < 0.3, (
        f"Position disagreement too large: median={median:.4f} m, "
        f"p90={p90:.4f} m, max={dmax:.4f} m"
    )


# ---- Tier 2: fast CPU/GPU guards --------------------------------------------


@pytest.mark.tier2
def test_forward_backward_finite():
    """Full forward+backward pass: loss and all gradients must be finite."""
    _skip_no_data()
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    cfg = _cfg()
    ds = GrandTourDataset(cfg, [MISSION], train=False)
    loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
    batch = next(iter(loader))

    device = torch.device("cuda")
    model = _Silent(cfg).to(device)
    model.train()

    opt = torch.optim.AdamW(model.model.parameters(), lr=1e-4)
    opt.zero_grad()

    color, depth_voxel, K_full, K, ext, path, target, weights, depth_t, depth_m = [
        x.to(device) for x in batch
    ]
    trav_map, _, pred_depth, _ = model.model(color, depth_voxel, K_full, K, ext, depth_t)
    trav_loss, _ = model.trav_criterion(path, trav_map, target, weights)
    depth_loss = model.depth_criterion(
        pred_depth,
        depth_t.view(-1, *depth_t.shape[2:]),
        depth_m.view(-1, *depth_m.shape[2:]),
    )
    loss = trav_loss + model.depth_weight * depth_loss

    print(
        f"\nForward pass: trav_loss={trav_loss.item():.4f}, "
        f"depth_loss={depth_loss.item():.4f}, total={loss.item():.4f}"
    )
    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"

    loss.backward()

    bad = [
        (n, "None" if p.grad is None else "inf/nan")
        for n, p in model.named_parameters()
        if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all())
    ]
    assert not bad, f"Non-finite/missing gradients in: {bad[:10]}"


@pytest.mark.tier2
def test_label_weight_integrity():
    """Labels in [0,1], no NaN; no training sample maps to an inf weight bin."""
    _skip_no_data()
    cfg = _cfg()
    ds = GrandTourDataset(cfg, [MISSION], train=False)

    for seg in ds.rosbags:
        trav = np.asarray(seg["traversability"])
        assert np.all(np.isfinite(trav)), "Non-finite traversability labels"
        assert trav.min() >= 0.0, f"Labels below 0: min={trav.min():.4f}"
        assert trav.max() <= 1.0, f"Labels above 1: max={trav.max():.4f}"

    weights = ds.weights
    n_inf = int(np.sum(np.isinf(weights)))
    finite_w = weights[np.isfinite(weights)]
    print(
        f"\nWeights: min={finite_w.min():.3f}, max={finite_w.max():.3f}, "
        f"n_inf={n_inf}/{weights.size} bins"
    )
    if n_inf > 0:
        print(
            "  NOTE: empty histogram bins -> inf weights exist. "
            "No current sample maps to them (narrow label range)."
        )

    assert not np.any(np.isnan(weights)), "NaN in weights"

    for seg in ds.rosbags:
        trav = np.asarray(seg["traversability"])
        idxs = np.digitize(trav, ds.bins[:-1]) - 1
        idxs = np.clip(idxs, 0, weights.shape[0] - 1)
        assert np.all(np.isfinite(weights[idxs[:, 0], 0])), (
            "Sample maps to inf weight bin (column 0)"
        )
        assert np.all(np.isfinite(weights[idxs[:, 1], 1])), (
            "Sample maps to inf weight bin (column 1)"
        )


@pytest.mark.tier2
def test_velocity_sanity():
    """|v| < 3 m/s and |ω| < 3 rad/s for >99% of steps (plausible ANYmal ranges)."""
    if not os.path.isdir(LABELS_ROOT):
        pytest.skip(f"LABELS_ROOT not available: {LABELS_ROOT}")

    missions = sorted(
        m
        for m in os.listdir(LABELS_ROOT)
        if os.path.isdir(os.path.join(LABELS_ROOT, m, "rhe"))
        and all(
            os.path.exists(os.path.join(LABELS_ROOT, m, "rhe", f))
            for f in ("rhe_states.npy", "rhe_timestamp.npy")
        )
    )
    if not missions:
        pytest.skip("No missions with rhe_states/timestamp")

    print()
    for m in missions:
        rhe_dir = os.path.join(LABELS_ROOT, m, "rhe")
        rhe_states = np.load(os.path.join(rhe_dir, "rhe_states.npy"))
        ts = np.load(os.path.join(rhe_dir, "rhe_timestamp.npy")).astype(np.float64)

        x, y, theta = rhe_states[:, 0], rhe_states[:, 1], rhe_states[:, 2]
        vx = np.gradient(x, ts)
        vy = np.gradient(y, ts)
        v = vx * np.cos(theta) + vy * np.sin(theta)
        omega = np.gradient(np.unwrap(theta), ts)

        v_ok = float(np.mean(np.abs(v) < 3.0))
        o_ok = float(np.mean(np.abs(omega) < 3.0))
        print(
            f"  {m}: |v|_max={np.abs(v).max():.2f} m/s ({v_ok * 100:.1f}% ok),  "
            f"|ω|_max={np.abs(omega).max():.2f} rad/s ({o_ok * 100:.1f}% ok)"
        )
        assert v_ok >= 0.99, f"{m}: {(1 - v_ok) * 100:.1f}% of v exceeds 3 m/s"
        assert o_ok >= 0.99, f"{m}: {(1 - o_ok) * 100:.1f}% of omega exceeds 3 rad/s"


@pytest.mark.tier2
def test_single_sample_overfit():
    """Model must drive trav_error < 0.05 on 4 samples (structural sanity)."""
    _skip_no_data()
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    SMALL_N = 4
    EPOCHS = 300

    pl.seed_everything(0, workers=True)
    cfg = _cfg()
    cfg.TRAINING.EPOCHS = EPOCHS

    ds = GrandTourDataset(cfg, [MISSION], train=True)
    subset = Subset(ds, list(range(min(SMALL_N, len(ds)))))
    loader = DataLoader(
        subset, batch_size=SMALL_N, shuffle=True, drop_last=True, num_workers=0
    )
    eval_loader = DataLoader(subset, batch_size=SMALL_N, shuffle=False, num_workers=0)

    device = torch.device("cuda")
    model = _Silent(cfg).to(device)
    model.train()

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,  # single GPU: 4-sample dataset can't survive DDP splitting
        max_epochs=EPOCHS,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
    )
    trainer.fit(model, loader)

    after = _mean_trav_error(model, eval_loader, device)
    print(f"\nSingle-set overfit trav_error (4 samples, {EPOCHS} epochs): {after:.4f}")
    assert after < 0.05, (
        f"Model failed to overfit 4 samples: error={after:.4f} (target < 0.05)"
    )


# ---- Tier 3: robustness -----------------------------------------------------


@pytest.mark.tier3
def test_flip_augmentation():
    """Horizontal flip must negate states[:,1] (y), [:,2] (θ), [:,4] (ω)."""
    _skip_no_data()
    cfg = _cfg()
    ds = GrandTourDataset(cfg, [MISSION], train=False)

    rosbag, data_idx = _locate(ds, 0)
    ts = [rosbag["image_timestamp"][data_idx + t] for t in range(ds.n_frames)]

    states_nf, _, _, _ = ds.get_states(rosbag, ts, np.eye(4), False)
    states_fl, _, _, _ = ds.get_states(rosbag, ts, np.eye(4), True)

    np.testing.assert_allclose(states_fl[:, 0], states_nf[:, 0], atol=1e-6)  # x
    np.testing.assert_allclose(states_fl[:, 1], -states_nf[:, 1], atol=1e-6)  # y
    np.testing.assert_allclose(states_fl[:, 2], -states_nf[:, 2], atol=1e-6)  # θ
    np.testing.assert_allclose(states_fl[:, 3], states_nf[:, 3], atol=1e-6)  # v
    np.testing.assert_allclose(states_fl[:, 4], -states_nf[:, 4], atol=1e-6)  # ω

    # With pose aug A the labels must be aug(flip(p)), matching the extrinsics
    # composition A @ flip(E) in __getitem__ (flip FIRST, then aug).
    theta = 0.3
    A = np.eye(4)
    A[:2, :2] = [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    A[:2, 3] = [0.4, -0.7]
    states_fa, _, _, _ = ds.get_states(rosbag, ts, A, True)
    mirrored = states_nf[:, :2] * [1, -1]
    expected_xy = mirrored @ A[:2, :2].T + A[:2, 3]
    np.testing.assert_allclose(states_fa[:, :2], expected_xy, atol=1e-6)
    np.testing.assert_allclose(states_fa[:, 2], -states_nf[:, 2] + theta, atol=1e-6)

    # Intrinsics: fliplr sends pixel u to (W-1)-u, so cx_flipped = (W-1) - cx.
    # W is the native depth width (K is native-res).
    depth = cv2.imread(rosbag["depth_image"][data_idx], -1)
    assert depth is not None, f"Cannot read depth: {rosbag['depth_image'][data_idx]}"
    W = depth.shape[1]
    K = rosbag["intrinsics"][data_idx].copy()
    K_fl = K.copy()
    K_fl[0, 2] = W - 1 - K_fl[0, 2]

    # Extrinsics mirror conjugation: projecting the mirrored path through the
    # flipped extrinsics must give the exact pixel mirror of the base projection.
    c2b = [rosbag["cam2base"][data_idx + t].copy() for t in range(ds.n_frames)]
    ext_nf = ds.get_extrinsics(rosbag, ts, [c.copy() for c in c2b], False)[-1]
    ext_fl = ds.get_extrinsics(rosbag, ts, [c.copy() for c in c2b], True)[-1]

    def proj(pts3, K, ext):
        cam = (pts3 - ext[:3, 3]) @ ext[:3, :3]
        keep = cam[:, 2] > 1e-3
        pix = cam[keep] @ K.T
        return pix[:, :2] / pix[:, 2:3], keep

    p3 = np.column_stack([states_nf[:, :2], np.zeros(len(states_nf))])
    uv0, keep0 = proj(p3, K, ext_nf)
    uv1, keep1 = proj(p3 * [1, -1, 1], K_fl, ext_fl)
    np.testing.assert_array_equal(keep0, keep1)
    np.testing.assert_allclose(uv1[:, 0], (W - 1) - uv0[:, 0], atol=1e-6)
    np.testing.assert_allclose(uv1[:, 1], uv0[:, 1], atol=1e-6)


@pytest.mark.tier3
def test_dataset_integrity_sweep():
    """All missions with label files have consistent array shapes and ≥1 segment."""
    if not os.path.isdir(DATA_ROOT):
        pytest.skip(f"DATA_ROOT not available: {DATA_ROOT}")

    required_extracted = ["cam2base.npy", "intrinsics.npy", "dlio_odom.npy"]
    required_labels = ["rhe_states.npy", "rhe_timestamp.npy", "rhe_trav_labels.npy"]
    print()
    for m in sorted(os.listdir(DATA_ROOT)):
        mdir = os.path.join(DATA_ROOT, m)
        if not os.path.isdir(mdir):
            continue
        rhe_dir = os.path.join(LABELS_ROOT, m, "rhe")
        missing = [f for f in required_extracted if not os.path.exists(os.path.join(mdir, f))]
        missing += [f for f in required_labels if not os.path.exists(os.path.join(rhe_dir, f))]
        if missing:
            print(f"  {m}: skip (missing {missing})")
            continue

        rhe_states = np.load(os.path.join(rhe_dir, "rhe_states.npy"))
        ts = np.load(os.path.join(rhe_dir, "rhe_timestamp.npy"))
        trav = np.load(os.path.join(rhe_dir, "rhe_trav_labels.npy"))

        assert rhe_states.ndim == 2 and rhe_states.shape[1] == 3, (
            f"{m}: rhe_states shape {rhe_states.shape}"
        )
        assert ts.ndim == 1, f"{m}: timestamp shape {ts.shape}"
        assert trav.ndim == 2 and trav.shape[1] >= 2, (
            f"{m}: trav_labels shape {trav.shape}"
        )
        N = len(ts)
        assert len(rhe_states) == N, (
            f"{m}: rhe_states len {len(rhe_states)} != ts len {N}"
        )
        assert len(trav) == N, f"{m}: trav len {len(trav)} != ts len {N}"

        has_images = os.path.exists(os.path.join(mdir, "image_manifest.npz"))
        has_depth = os.path.exists(os.path.join(mdir, "depth_manifest.npz"))

        n_segs = "–"
        if has_images and has_depth:
            cfg = _cfg()
            ds = GrandTourDataset(cfg, [m], train=False)
            n_segs = len(ds.rosbags)
            assert n_segs >= 1, f"{m}: no valid segments after loading"

        print(
            f"  {m}: N={N}, images={'y' if has_images else 'n'}, "
            f"depth={'y' if has_depth else 'n'}, segs={n_segs}"
        )


@pytest.mark.tier3
def test_depth_health():
    """Valid-pixel and in-dbound fraction must exceed floor thresholds."""
    if not os.path.isdir(DATA_ROOT):
        pytest.skip(f"DATA_ROOT not available: {DATA_ROOT}")

    missions = [
        m
        for m in sorted(os.listdir(DATA_ROOT))
        if os.path.isdir(os.path.join(DATA_ROOT, m))
        and os.path.exists(os.path.join(DATA_ROOT, m, "depth_manifest.npz"))
    ]
    if not missions:
        pytest.skip("No missions with depth_manifest.npz")

    dmin_mm = 300  # 0.3 m
    dmax_mm = 8000  # 8.0 m

    print()
    for m in missions:
        mdir = os.path.join(DATA_ROOT, m)
        dm = np.load(os.path.join(mdir, "depth_manifest.npz"))
        depth_arr = dm["depth"]

        step = max(1, len(depth_arr) // 100)
        sample = depth_arr[::step]

        valid_fracs, useful_fracs = [], []
        for row in sample:
            path = os.path.join(mdir, row["path"])
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            img = img[BORDER_CROP:].astype(np.float32)
            total = img.size
            valid_fracs.append(np.sum(img > 0) / total)
            useful_fracs.append(np.sum((img >= dmin_mm) & (img <= dmax_mm)) / total)

        mean_valid = float(np.mean(valid_fracs))
        mean_useful = float(np.mean(useful_fracs))
        print(
            f"  {m}: valid={mean_valid:.1%}, in_dbound={mean_useful:.1%} "
            f"({len(valid_fracs)}/{len(depth_arr)} frames sampled)"
        )
        assert mean_valid > 0.3, f"{m}: valid fraction {mean_valid:.1%} < 30%"
        assert mean_useful > 0.25, f"{m}: in-dbound fraction {mean_useful:.1%} < 25%"
