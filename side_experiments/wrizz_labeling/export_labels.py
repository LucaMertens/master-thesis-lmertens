"""labels.jsonl -> per-mission CSR index of propagated observations, keyed by ts_ns.

Run: uv run python side_experiments/wrizz_labeling/export_labels.py"""

import argparse
import collections
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from traversability_network.adaptations import propagate as P

REGION_K = 8  # default world points kept per segment, sampled at training time (--region-k)


def merge_amendments(records):
    """Apply amendments (keyed by answer uuid) and drop undone answers; last answer per pair wins."""
    by_uuid = {r["uuid"]: r for r in common.live_records(records)}
    for r in records:
        if not r.get("amend"):
            continue
        b = by_uuid.get(r.get("target"))
        if b is None:
            continue
        for side, key in (("a", "prop_a"), ("b", "prop_b")):
            if key in r:
                b[side]["prop"] = r[key]
        if "flags" in r:
            b["flags"] = sorted(set(b.get("flags", [])) | set(r["flags"]))
        # flags accumulate, so clearing needs its own key
        if "unflags" in r:
            b["flags"] = sorted(set(b.get("flags", [])) - set(r["unflags"]))
    latest = {}
    for r in by_uuid.values():
        latest[r["id"]] = r
    return list(latest.values())


def prompted_mask(e):
    """The click-prompted region for an endpoint, or None. Decodes sam.rle."""
    code = e.get("region_rle")
    if not code:
        return None
    import sam as sam_mod
    return sam_mod.unrle(code)


def region_points(idmap, seg, anchor, frame, geom, radius_m, k=REGION_K, rng=None):
    """Up to k world points sampled inside one SAM segment.

    ``radius_m`` >= REGION_FULL_M takes the whole segment, below it clips to that radius
    around the anchor, 0 drops the region and keeps the bare point.
    """
    if radius_m <= 0:
        return np.zeros((0, 3))
    # seg is a SAM id or a boolean mask from a click-prompted region
    ys, xs = np.nonzero(seg if isinstance(seg, np.ndarray) else (idmap == seg))
    clip = radius_m < P.REGION_FULL_M
    if clip:
        cx, cy = common.display_pixel(anchor["uv"])
        rpx = radius_m * (geom.K[0, 0] / common.SCALE) / anchor["z"]
        near = (xs - cx) ** 2 + (ys - cy) ** 2 <= rpx * rpx
        xs, ys = xs[near], ys[near]
    if not len(xs):
        return np.zeros((0, 3))

    rng = rng or np.random.default_rng(0)
    pick = rng.choice(len(xs), min(k * 3, len(xs)), replace=False)
    origin = np.asarray(anchor["p_world"])
    out = []
    for i in pick:
        a = P.anchor_from_pixel((xs[i] * common.SCALE, ys[i] * common.SCALE), frame, geom)
        if a["status"] != "ok":
            continue
        # the pixel disc only approximates the world radius
        if clip and np.linalg.norm(np.asarray(a["p_world"]) - origin) > radius_m:
            continue
        out.append(a["p_world"])
        if len(out) >= k:
            break
    return np.array(out) if out else np.zeros((0, 3))


def overlapping(pa, pb, ra, rb):
    """Whether two region discs overlap, in world metres. Same test as the UI."""
    if ra <= 0 or rb <= 0:
        return False
    if ra >= P.REGION_FULL_M or rb >= P.REGION_FULL_M:
        return True
    return float(np.linalg.norm(np.asarray(pa) - np.asarray(pb))) < ra + rb


def shared_regions(labels, out_dir, radius_of, world_of):
    """(label_id, side) whose region is also the partner's.

    Intra pairs often sit on one surface. Both points still propagate; only the region
    expansion is dropped, since it would call one surface both better and worse than itself.
    """
    idmaps, drop = {}, set()
    for r in labels:
        a, b = r["a"], r["b"]
        if (a["mission"], a["ts_ns"]) != (b["mission"], b["ts_ns"]) or r["t"] == 0:
            continue
        # recomputed anchors, not the stored p_world
        pa, pb = world_of(r["id"], "a"), world_of(r["id"], "b")
        if pa is None or pb is None:
            continue
        path = os.path.join(out_dir, "segments", f"{a['mission']}_{a['ts_ns']}.png")
        if path not in idmaps:
            idmaps[path] = common.load_idmap(path)
        idmap = idmaps[path]
        if idmap is None:
            continue
        sa, sb = common.segment_at(idmap, a["uv"]), common.segment_at(idmap, b["uv"])
        if sa and sa == sb and overlapping(pa, pb, radius_of(a), radius_of(b)):
            drop |= {(r["id"], "a"), (r["id"], "b")}
    return drop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="side_experiments/wrizz_labeling/output")
    ap.add_argument("--cfg", default=common.CFG_FILE)
    ap.add_argument("--index-root", default=None)
    ap.add_argument("--no-regions", action="store_true")
    ap.add_argument("--region-k", type=int, default=REGION_K,
                    help="world points kept per region")
    ap.add_argument("--no-planar", action="store_true",
                    help="keep whole segments instead of cutting them at the surface break")
    args = ap.parse_args()

    cfg = common.load_cfg(args.cfg)
    index_root = args.index_root or os.path.join(args.out, "index")
    os.makedirs(index_root, exist_ok=True)

    n_planar = 0
    labels = merge_amendments(common.read_jsonl(os.path.join(args.out, "labels.jsonl")))
    labels = [r for r in labels if r["t"] is not None]
    # a pair goes entirely if either endpoint is excluded
    n_before = len(labels)
    labels = [r for r in labels
              if not any(common.excluded(r[s]["mission"], r[s]["ts_ns"]) for s in ("a", "b"))]
    if n_before != len(labels):
        print(f"dropped {n_before - len(labels)} answers inside an excluded stretch "
              f"({common.EXCLUDE_RANGES})")
    # anchor and propagation gates both invalidate the index
    prop_cfg = {"window": P.WINDOW, "z_max_anchor": P.Z_MAX_ANCHOR,
                "z_max_target": P.Z_MAX_TARGET, "z_min": P.Z_MIN,
                "resid": [P.RESID_ABS, P.RESID_FRAC], "spread": [P.SPREAD_ABS, P.SPREAD_FRAC],
                "anchor_win": P.ANCHOR_WIN, "min_valid": P.MIN_VALID, "border": P.BORDER,
                "border_crop": P.BORDER_CROP, "baseline": P.MIN_BASELINE,
                "max_obs": P.MAX_OBS, "min_side": P.MIN_SIDE, "region_k": args.region_k,
                "region_full_m": P.REGION_FULL_M}
    sha = hashlib.sha1(json.dumps(prop_cfg, sort_keys=True).encode()).hexdigest()[:12]

    # A label may enter the loss only if both endpoints are in id_train
    # (heldout or OOD endpoints would contaminate selection and OOD evaluation)
    train_missions = set(cfg.DATASET.TRAIN_MISSIONS)

    def trainable(r):
        return all(r[s]["mission"] in train_missions for s in ("a", "b"))

    def radius_of(e):
        # 0 means point only, so test None explicitly
        r = e.get("region_m")
        return P.REGION_FULL_M if r is None else float(r)

    # grouped by mission so each MissionGeometry is built once
    by_mission = {}
    for r in labels:
        for side in ("a", "b"):
            if r[side].get("prop"):
                by_mission.setdefault(r[side]["mission"], []).append((r, side))

    # Recompute all anchors from the click; the stored p_world is not trusted, so gate fixes reach old labels
    geoms, anchor_of, stale = {}, {}, 0
    for mission, entries in sorted(by_mission.items()):
        geoms[mission] = geom = P.MissionGeometry(
            os.path.join(cfg.DATASET.DATA_ROOT, mission), cfg.DATASET.DEPTH_CACHE_ROOT, cache=8)
        for r, side in entries:
            e = r[side]
            a = P.anchor_from_pixel(e["uv"], geom.frame_index(e["ts_ns"]), geom, mode="auto")
            stale += a["status"] != e.get("status")
            if a["status"] == "ok":
                anchor_of[(r["id"], side)] = a

    def world_of(label_id, side):
        a = anchor_of.get((label_id, side))
        return a["p_world"] if a else None

    no_region = set() if args.no_regions else shared_regions(
        labels, args.out, radius_of, world_of)

    total_obs = 0
    for mission, entries in sorted(by_mission.items()):
        geom = geoms[mission]
        anchors, frames, keep = [], [], []
        for r, side in entries:
            a = anchor_of.get((r["id"], side))
            if a is None:
                continue
            keep.append((r, side))
            anchors.append(a)
            frames.append(geom.frame_index(r[side]["ts_ns"]))
        if not keep:
            continue

        p_world = np.array([a["p_world"] for a in anchors])
        offsets, obs_f, obs_uv, obs_z, obs_resid = P.propagate_batch(p_world, frames, geom)

        regions = np.zeros((len(keep), args.region_k, 3), np.float32)
        n_region = np.zeros(len(keep), np.int16)
        if not args.no_regions:
            rng = np.random.default_rng(0)
            for i, ((r, side), f) in enumerate(zip(keep, frames)):
                e = r[side]
                if (r["id"], side) in no_region:
                    continue
                path = os.path.join(args.out, "segments", f"{mission}_{e['ts_ns']}.png")
                idmap = common.load_idmap(path)
                if idmap is None:
                    continue
                seg = common.segment_at(idmap, e["uv"])
                if not seg:
                    continue
                # cut the segment at the surface break; a radius cannot follow a narrow trail
                # (halves driven-path share of a region, 0.66 -> 0.33, keeping median 89%)
                region = prompted_mask(e) if e.get("region_rle") else seg
                if not args.no_planar and not isinstance(region, np.ndarray):
                    cut = P.planar_mask(idmap, region, anchors[i], f, geom, common.SCALE)
                    if cut is not None and cut.any():
                        region = cut
                        n_planar += 1
                pts = region_points(idmap, region, anchors[i], f, geom, radius_of(e),
                                    k=args.region_k, rng=rng)
                n_region[i] = len(pts)
                regions[i, : len(pts)] = pts

        np.savez_compressed(
            os.path.join(index_root, f"{mission}.npz"),
            label_id=np.array([r["id"] for r, _ in keep]),
            side=np.array([s for _, s in keep]),
            # lets consumers apply common.EVAL_SOURCES without the JSONL
            pair_source=np.array([r["pair_source"] for r, _ in keep]),
            seg_rel=np.array([r.get("seg_rel") or "" for r, _ in keep]),
            hdr_eligible=np.array([common.hdr_eligible(r) for r, _ in keep], bool),
            split=np.array([r["split"] for r, _ in keep]),
            kind=np.array([r["kind"] for r, _ in keep]),
            t=np.array([r["t"] for r, _ in keep], np.int8),
            # false = eval-only
            trainable=np.array([trainable(r) for r, _ in keep], bool),
            train_missions=np.array(sorted(train_missions)),
            anchor_ts_ns=np.array([r[s]["ts_ns"] for r, s in keep], np.int64),
            anchor_uv=np.array([r[s]["uv"] for r, s in keep], np.float32),
            p_world=p_world.astype(np.float32),
            height=np.array([a["height_over_ground"] for a in anchors], np.float32),
            offsets=offsets,
            obs_ts_ns=geom.ts_ns[obs_f],
            obs_uv=obs_uv, obs_z=obs_z, obs_resid=obs_resid,
            region_pts=regions, n_region=n_region,
            region_m=np.array([radius_of(r[sd]) for r, sd in keep], np.float32),
            prop_config_sha=sha, prop_config=json.dumps(prop_cfg))
        n = np.diff(offsets)
        total_obs += len(obs_f)
        print(f"{mission}: {len(keep)}/{len(entries)} anchors, {len(obs_f)} obs "
              f"(median {int(np.median(n))}/anchor, {(n == 0).sum()} dropped), "
              f"regions median {int(np.median(n_region))}")

    n_train = sum(trainable(r) for r in labels)
    leaks = collections.Counter(
        r[s]["mission"] for r in labels if not trainable(r) for s in ("a", "b")
        if r[s]["mission"] not in train_missions)
    print(f"{len(labels)} labels, {total_obs} observations -> {index_root} (sha {sha})")
    print(f"  trainable {n_train}, eval-only {len(labels) - n_train} "
          f"(endpoints outside id_train: {dict(leaks)})")
    print(f"  {n_planar} regions cut at the surface break "
          f"(--no-planar keeps whole segments)")
    print(f"  {len(no_region) // 2} intra pairs share a region, region expansion "
          f"suppressed (their points still propagate)")
    if stale:
        print(f"  WARNING: {stale} endpoints changed anchor status since they were "
              f"labelled; the stored p_world was stale and has been recomputed")


if __name__ == "__main__":
    main()
