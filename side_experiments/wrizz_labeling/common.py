"""Shared helpers for the labeling tool: splits, frame index, k-center selection, id-maps."""

import functools
import json
import os

import cv2

import numpy as np

from shared_utils.paths import LABELS_ROOT
from shared_utils.tiers import time_ranges
from ood_detection.common.splits import TIER_MISSIONS
from traversability_network.config import get_cfg

# Annotator display size; native is 3x
SIZE = (480, 360)
NATIVE = (1440, 1080)
SCALE = 3

# F2 is the reported fold; grandtour.yaml splits CYN, HOEB and LMB recordings across train/eval
CFG_FILE = "training/traversability_network/configs/grandtour_baseline_f2.yaml"
CAMERA = "front_center"

# Only uniformly drawn pairs may enter HDR; rhe_overlap and hunt over-sample on purpose
EVAL_SOURCES = ("random",)


def hdr_eligible(r):
    """Whether an answer may enter an HDR estimate.

    Same-segment pairs (annotator check, foregone tie) and pairs shown with the driven
    path (not a blind judgement) are excluded.
    """
    return (r.get("pair_source") in EVAL_SOURCES and r.get("seg_rel") != "same"
            and "path_shown" not in (r.get("flags") or []))


# Hunt pairs may only be placed here; elsewhere they would bias the metric
HUNT_SPLITS = ("id_train",)


# Missions merge for keyframe selection below this mean nearest-neighbour cosine distance.
# On the F2 train missions CYN-1/CYN-2 sit at 0.0115 and the next pair at 0.0218.
# Measured, not by name prefix: LMB-2 is closer to HOEB and ALB than to LMB-1.
GROUP_MAX_DIST = 0.018


def pair_distance(A, B, sample=600):
    """Symmetric coverage distance: mean nearest-frame distance, worse direction."""
    a = A[np.linspace(0, len(A) - 1, min(sample, len(A))).astype(int)]
    b = B[np.linspace(0, len(B) - 1, min(sample, len(B))).astype(int)]
    d = 1 - a @ b.T
    return min(float(d.min(1).mean()), float(d.min(0).mean()))


def redundant_groups(feats, thresh=GROUP_MAX_DIST):
    """Group missions that cover the same ground; keyframes are picked per group."""
    missions = list(feats)
    parent = {m: m for m in missions}

    def find(m):
        while parent[m] != m:
            parent[m] = parent[parent[m]]
            m = parent[m]
        return m

    for i, a in enumerate(missions):
        for b in missions[i + 1:]:
            if pair_distance(feats[a], feats[b]) < thresh:
                parent[find(a)] = find(b)
    groups = {}
    for m in missions:
        groups.setdefault(find(m), []).append(m)
    return {"+".join(sorted(v)): sorted(v) for v in groups.values()}


# Unix-second stretches not to label. RIV-1: operator carries the robot.
EXCLUDE_RANGES = {"RIV-1": [(1730735917.317, 1730735983.317)]}


def excluded(mission, ts_ns):
    t = ts_ns / 1e9
    return any(lo <= t <= hi for lo, hi in EXCLUDE_RANGES.get(mission, ()))


def keep_mask(mission, ts_ns):
    """Mask over an array of ts_ns, False inside an excluded stretch."""
    t = np.asarray(ts_ns, np.float64) / 1e9
    keep = np.ones(len(t), bool)
    for lo, hi in EXCLUDE_RANGES.get(mission, ()):
        keep &= (t < lo) | (t > hi)
    return keep


@functools.cache
def _spans(split):
    return time_ranges(split)


def span_mask(split, mission, ts_ns):
    """Mask over ts_ns, False outside the tiers.json spans of this split (no spans = all pass)."""
    t = np.asarray(ts_ns, np.float64) / 1e9
    spans = _spans(split).get(mission)
    if not spans:
        return np.ones(len(t), bool)
    keep = np.zeros(len(t), bool)
    for lo, hi in spans:
        keep |= (t >= lo) & (t < (np.inf if hi is None else hi))
    return keep


@functools.cache
def labeled_range(mission):
    """First and last RHE timestamp. Cameras outlast the labels, and a null span end would reach that tail."""
    ts = np.load(LABELS_ROOT / mission / "rhe" / "rhe_timestamp.npy", mmap_mode="r")
    return float(ts[0]), float(ts[-1])


def frame_mask(split, mission, ts_ns):
    """Frames a split may draw from: in its tier spans, the labeled range, and not excluded."""
    lo, hi = labeled_range(mission)
    t = np.asarray(ts_ns, np.float64) / 1e9
    return (keep_mask(mission, ts_ns) & span_mask(split, mission, ts_ns)
            & (t >= lo) & (t <= hi))


def load_cfg(cfg_file=CFG_FILE):
    return get_cfg(cfg_file)


def split_missions(cfg, split):
    """Missions in a labeling split; id_* come from the config's TRAIN/VALID_MISSIONS.

    Not ood_detection.all_tiers, which asserts the original ID/OOD split.
    """
    tiers = {"id_train": list(cfg.DATASET.TRAIN_MISSIONS),
             "id_heldout": list(cfg.DATASET.VALID_MISSIONS), **TIER_MISSIONS}
    if split not in tiers:
        raise SystemExit(f"unknown split {split!r}, have {sorted(tiers)}")
    overlap = set(tiers["id_train"]) & set(tiers["id_heldout"])
    if overlap:
        raise SystemExit(f"{sorted(overlap)} is in both train and valid; a within-mission "
                         "fold needs time-range aware selection, not whole missions")
    return tiers[split]


def mission_frames(cfg, mission, native=False):
    """(ts_ns sorted, absolute image paths). ``native`` skips the resized copies (too soft for display)."""
    data_root = cfg.DATASET.DATA_ROOT
    fc = np.load(os.path.join(data_root, mission, "image_manifest.npz"))[CAMERA]
    order = np.argsort(fc["ts_ns"].astype(np.int64))
    root = os.path.join(data_root, mission)
    if not native:
        resized = os.path.join(cfg.DATASET.RESIZED_ROOT, mission)
        if os.path.isdir(resized):
            root = resized
    paths = [os.path.join(root, p) for p in fc["path"][order]]
    return fc["ts_ns"].astype(np.int64)[order], paths


def pooled_features(cache_dir):
    """(ts_ns, L2-normalized mean-pooled DINO features (N, D)) in cache row order."""
    meta = np.load(os.path.join(cache_dir, "meta.npz"), allow_pickle=True)
    feats = np.load(os.path.join(cache_dir, "features.npy"), mmap_mode="r")
    pooled = np.empty((feats.shape[0], feats.shape[1]), np.float32)
    for lo in range(0, feats.shape[0], 256):
        pooled[lo : lo + 256] = feats[lo : lo + 256].astype(np.float32).mean(axis=(2, 3))
    pooled /= np.linalg.norm(pooled, axis=1, keepdims=True) + 1e-8
    return meta["ts_ns"].astype(np.int64), pooled


# Keyframes closer than this are the same photograph (2 s = 1.2 m at 0.62 m/s).
# k-center ignores time and otherwise picks consecutive frames (RIV-1: 118 of 216 within 1 s).
MIN_DT_S = 2.0


def k_center(X, k, seeds=None, ts=None, min_dt=MIN_DT_S):
    """Greedy k-center (cosine distance) row indices, seeded from the mean.

    Returns (picks, radii); radii[j] is the covering radius before pick j. Picks are
    prefix-consistent, so budget can be allocated after the curves are known.

    ``seeds`` are already-labeled rows: not returned, no budget used, but they lower the
    radius so picks go to new frames.
    """
    if seeds is not None and len(seeds):
        d = np.full(len(X), np.inf, np.float32)
        for i in seeds:
            d = np.minimum(d, 1 - np.einsum("ij,j->i", X, X[i]))
    else:
        center = X.mean(0)
        center /= np.linalg.norm(center) + 1e-8
        # einsum: ``X @ v`` takes a BLAS path that is ~250x slower here
        d = 1 - np.einsum("ij,j->i", X, center)
    # ts in ns
    gap = min_dt * 1e9 if ts is not None and min_dt else 0
    live = np.ones(len(X), bool)
    # neighbours of a seeded frame are duplicates too
    if gap and seeds is not None:
        for i in seeds:
            live &= np.abs(ts - ts[i]) >= gap
    picks, radii = [], []
    for _ in range(min(k, len(X))):
        i = int(np.argmax(np.where(live, d, -1.0)))
        if not live[i]:
            break  # every remaining frame is within min_dt of one already picked
        radii.append(float(d[i]))
        picks.append(i)
        d = np.minimum(d, 1 - np.einsum("ij,j->i", X, X[i]))
        if gap:
            live &= np.abs(ts - ts[i]) >= gap
    return picks, np.array(radii)


def allocate(radii, budget, kmin=20, max_share=1.3):
    """Per-mission keyframe counts under a total budget, equalizing covering radius.

    Uncapped, one mission took 41-59% of a split, so each is capped at ``max_share``
    times its uniform share and the surplus goes to the rest.
    """
    def counts(target, pool, budget):
        out = {}
        for m in pool:
            r = radii[m]
            k = int(np.argmax(r <= target)) if (r <= target).any() else len(r)
            out[m] = min(len(r), max(kmin, k))
        return out

    def solve(pool, budget):
        lo = min(radii[m].min() for m in pool)
        hi = max(radii[m].max() for m in pool)
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if sum(counts(mid, pool, budget).values()) > budget:
                lo = mid
            else:
                hi = mid
        return counts(hi, pool, budget), hi

    cap = max(kmin, int(max_share * budget / len(radii)))
    pool, fixed, left = list(radii), {}, budget
    for _ in range(len(radii)):
        got, radius = solve(pool, left)
        over = [m for m in pool if got[m] > cap]
        if not over:
            fixed.update(got)
            return fixed, radius
        for m in over:
            fixed[m] = min(cap, len(radii[m]))
            left -= fixed[m]
        pool = [m for m in pool if m not in fixed]
        if not pool:
            return fixed, radius
    fixed.update(got)
    return fixed, radius


def rle(mask):
    """Row-major run-length encoding of a binary mask: [w, h, run, run, ...], first run is 0s."""
    flat = np.asarray(mask, bool).ravel()
    runs = np.diff(np.flatnonzero(np.concatenate(([True], flat[1:] != flat[:-1], [True]))))
    out = runs.tolist()
    if flat[0]:
        out = [0] + out
    return [int(mask.shape[1]), int(mask.shape[0])] + [int(x) for x in out]


def unrle(code):
    """Inverse of rle."""
    w, h = code[0], code[1]
    flat = np.zeros(w * h, bool)
    i, val = 0, False
    for n in code[2:]:
        if val:
            flat[i : i + n] = True
        i += n
        val = not val
    return flat.reshape(h, w)


def encode_idmap(idmap):
    """uint32 segment ids packed into an RGB PNG (cv2 writes BGR)."""
    rgb = np.stack([idmap & 255, (idmap >> 8) & 255, (idmap >> 16) & 255], -1).astype(np.uint8)
    return rgb[..., ::-1]


def load_idmap(path):
    """Segment id map at SIZE, or None. Some maps on disk are 960x720, so resize on load."""
    # check first: cv2 warns per frame on missing files
    if not os.path.exists(path):
        return None
    img = cv2.imread(path)
    if img is None:
        return None
    idmap = decode_idmap(img)
    if (idmap.shape[1], idmap.shape[0]) != SIZE:
        # nearest only, ids are labels
        idmap = cv2.resize(idmap.astype(np.int32), SIZE,
                           interpolation=cv2.INTER_NEAREST).astype(np.uint32)
    return idmap


def decode_idmap(bgr):
    b, g, r = bgr[..., 0].astype(np.uint32), bgr[..., 1].astype(np.uint32), bgr[..., 2].astype(np.uint32)
    return r | (g << 8) | (b << 16)


def display_pixel(uv):
    """Native uv to display pixel. Must round like the UI."""
    return int(round(uv[0] / SCALE)), int(round(uv[1] / SCALE))


def segment_at(idmap, uv):
    """SAM segment id under a native-resolution click, 0 outside the map."""
    x, y = display_pixel(uv)
    h, w = idmap.shape
    return int(idmap[y, x]) if 0 <= x < w and 0 <= y < h else 0


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def live_records(records):
    """Committed answers: amendments and tombstones removed, undone answers dropped."""
    undone = {r["target"] for r in records if r.get("undo")}
    return [r for r in records
            if not r.get("amend") and not r.get("undo") and r["uuid"] not in undone]


def append_jsonl(path, record):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()
        os.fsync(f.fileno())
