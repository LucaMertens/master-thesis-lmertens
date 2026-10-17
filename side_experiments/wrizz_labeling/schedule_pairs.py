"""Generate the pair queue for a split: one intra pair per frame, one cross pair per two (W-RIZZ).

Coordinates are native pixels.

Run: uv run python side_experiments/wrizz_labeling/schedule_pairs.py --split id_heldout"""

import argparse
import collections
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
import rhe as rhe_mod
from traversability_network.adaptations.propagate import (
    BORDER_CROP, MissionGeometry, anchor_from_pixel,
)

W, H = common.NATIVE
# Paper: intra points at least 5% of min(W, H) apart
MIN_SEP = 0.05 * min(W, H)
# Share of intra pairs placed inside one segment (annotator check). Pixel separation alone is
# not enough: segments cover a median 37% of an image, and 56% of 366 intra pairs shared one.
SAME_SEG_FRAC = 0.10
# Minimum world distance between scheduled points in a mission (two points in one SAM mask ask the same question)
MIN_WORLD_M = 0.5
BORDER = 24
# Paper: 90% of points in the lower half
LOWER_FRAC = 0.9
# Re-placements tried before giving up on an endpoint
ANCHOR_TRIES = 24


def sample_point(rng):
    u = rng.integers(BORDER, W - BORDER)
    v0 = H // 2 if rng.random() < LOWER_FRAC else BORDER_CROP + BORDER
    v1 = H - BORDER
    return float(u), float(rng.integers(v0, v1))


def segments(out_dir, cache={}):
    """Display-resolution SAM id map per frame name, None when it was never computed."""
    def get(name):
        if name not in cache:
            cache[name] = common.load_idmap(os.path.join(out_dir, "segments", f"{name}.png"))
        return cache[name]
    return get


def seg_ok(idmap, a, b, want_same):
    if idmap is None:
        return True
    same = common.segment_at(idmap, a) == common.segment_at(idmap, b)
    return same == want_same


def sample_intra(rng, idmap, want_same, tries=64):
    """Two separated points in one frame with the requested segment relation, or None."""
    for _ in range(tries):
        a = sample_point(rng)
        for _ in range(8):
            b = sample_point(rng)
            if np.hypot(b[0] - a[0], b[1] - a[1]) >= MIN_SEP and seg_ok(idmap, a, b, want_same):
                return a, b
    return None


def endpoint(frame, uv, rhe_index=None):
    e = {"frame": frame["name"], "mission": frame["mission"],
         "ts_ns": frame["ts_ns"], "uv": [round(uv[0], 1), round(uv[1], 1)]}
    if rhe_index is not None:
        e["rhe_index"] = int(rhe_index)
    return e


def rhe_points(cfg, frames, rng, want, geom_of, seg_of):
    """Intra pairs whose endpoints are projected RHE path states (the only human-vs-RHE samples)."""
    out = []
    by_mission = {}
    for f in frames:
        by_mission.setdefault(f["mission"], []).append(f)
    for mission, mframes in by_mission.items():
        quota = round(want * len(mframes) / len(frames))
        if quota <= 0:
            continue
        mission_dir = os.path.join(cfg.DATASET.DATA_ROOT, mission)
        geom = geom_of(mission)
        labels = rhe_mod.RheLabels(cfg.DATASET.LABELS_ROOT, mission, mission_dir)
        rng.shuffle(mframes)
        for f in mframes:
            if quota <= 0:
                break
            idx, uv = labels.visible_states(geom, geom.frame_index(f["ts_ns"]))
            if len(idx) < 2:
                continue
            # states lie along one path, so prefer a different-segment draw but accept any of 16
            idmap = seg_of(f["name"])
            best = None
            for _ in range(16):
                i, j = rng.choice(len(idx), 2, replace=False)
                if np.hypot(*(uv[i] - uv[j])) < MIN_SEP:
                    continue
                best = best or (i, j)
                if seg_ok(idmap, uv[i], uv[j], False):
                    best = (i, j)
                    break
            if best:
                i, j = best
                out.append({"kind": "intra", "pair_source": "rhe_overlap",
                            "seg_rel": "diff" if seg_ok(idmap, uv[i], uv[j], False) else "same",
                            "a": endpoint(f, uv[i], idx[i]),
                            "b": endpoint(f, uv[j], idx[j])})
                quota -= 1
    return out


def anchor_points(cfg, frames, rng, want, geom_of):
    """One driven-path endpoint per frame, for cross pairs against OOD points.

    Pinning one side to the path cuts the cross answer sd from 0.82 to 0.35. RHE only selects; the human answer is the label.
    """
    out, by_mission = [], {}
    for f in frames:
        by_mission.setdefault(f["mission"], []).append(f)
    for mission, mframes in sorted(by_mission.items()):
        # ceil, so the queue is not short of anchors
        quota = -(-want * len(mframes) // len(frames))
        geom = geom_of(mission)
        labels = rhe_mod.RheLabels(cfg.DATASET.LABELS_ROOT, mission,
                                   os.path.join(cfg.DATASET.DATA_ROOT, mission))
        rng.shuffle(mframes)
        for f in mframes:
            if quota <= 0:
                break
            idx, uv = labels.visible_states(geom, geom.frame_index(f["ts_ns"]))
            if not len(idx):
                continue
            k = int(rng.integers(len(idx)))
            out.append(endpoint(f, uv[k], idx[k]))
            quota -= 1
    rng.shuffle(out)
    return out[:want]


def far_enough(placed, p, r=MIN_WORLD_M):
    """Whether world point p is at least r from every point already scheduled here."""
    if not placed:
        return True
    return bool(np.all(np.linalg.norm(np.asarray(placed) - np.asarray(p), axis=1) >= r))


def repair(pairs, rng, geom_of, seg_of, placed=None):
    """Re-place or drop endpoints that cannot be anchored or sit within MIN_WORLD_M of another point.

    Image-space sampling clusters on the narrow walked ribbon (CYN-2: 22% of points within
    25 cm of another). ``placed`` carries already-answered points in. rhe_overlap endpoints
    are path states, so they are dropped rather than moved. Visits endpoints in frame
    order so each depth image is read once per pass.
    """
    placed = collections.defaultdict(list) if placed is None else placed
    dead, moved, crowded = set(), 0, 0
    for side in ("a", "b"):
        other = "b" if side == "a" else "a"
        order = sorted(range(len(pairs)),
                       key=lambda i: (pairs[i][side]["mission"], pairs[i][side]["ts_ns"]))
        for i in order:
            p = pairs[i]
            if i in dead or p["pair_source"] == "hunt":
                continue
            e = p[side]
            geom = geom_of(e["mission"])
            frame = geom.frame_index(e["ts_ns"])
            intra = p["kind"] == "intra"
            sep = MIN_SEP if intra else 0.0
            fixed = p[other]["uv"]
            # keep the pair's segment relation when re-placing
            idmap = seg_of(e["frame"]) if intra else None
            want_same = p.get("seg_rel") == "same"
            # path-state points cannot move, but still register so random points give way
            on_path = "rhe_index" in e
            uv, tries = tuple(e["uv"]), 1 if on_path else ANCHOR_TRIES
            for k in range(tries):
                if (np.hypot(uv[0] - fixed[0], uv[1] - fixed[1]) >= sep
                        and seg_ok(idmap, uv, fixed, want_same)):
                    a = anchor_from_pixel(uv, frame, geom)
                    if a["status"] == "ok" and (on_path or far_enough(placed[e["mission"]],
                                                                      a["p_world"])):
                        e["uv"] = [round(uv[0], 1), round(uv[1], 1)]
                        placed[e["mission"]].append(a["p_world"])
                        moved += k > 0
                        break
                    crowded += a["status"] == "ok"
                uv = sample_point(rng)
            else:
                dead.add(i)
    return [p for i, p in enumerate(pairs) if i not in dead], len(dead), moved, crowded


def cross_couples(frames, rng, cross_mission_frac):
    """Couple frames into cross pairs without replacement; a share straddles missions.

    The cross-mission count is fixed up front and draws come from the largest pools, so
    missions drain evenly and the requested share is met.
    """
    pools = {}
    for i, f in enumerate(frames):
        pools.setdefault(f["mission"], []).append(i)
    for p in pools.values():
        rng.shuffle(p)

    total = sum(len(p) for p in pools.values()) // 2
    want_cross, made_cross, out = round(cross_mission_frac * total), 0, []
    for k in range(total):
        # largest first, ties broken at random, so no pool is left unpairable
        order = sorted((m for m, p in pools.items() if p),
                       key=lambda m: (-len(pools[m]), rng.random()))
        both = [m for m in order if len(pools[m]) >= 2]
        need, left = want_cross - made_cross, total - k
        cross = len(order) >= 2 and (not both or need >= left or rng.random() < need / left)
        if cross:
            m1, m2 = order[0], order[1]
            made_cross += 1
        elif both:
            m1 = m2 = both[0]
        else:
            break
        out.append((pools[m1].pop(), pools[m2].pop()))
    return out


def pair_id(split, p):
    """Content-derived id, stable across scheduler reruns."""
    key = json.dumps([p["kind"], p["pair_source"], p["a"], p["b"], p.get("dup_of")],
                     sort_keys=True)
    return f"{split}:{hashlib.sha1(key.encode()).hexdigest()[:12]}"


def from_record(r):
    """Rebuild the queue entry from the label record (not from pairs_<split>.jsonl, which gets overwritten)."""
    p = {"kind": r["kind"], "pair_source": r["pair_source"], "dup_of": r["dup_of"]}
    for s in ("a", "b"):
        e = r[s]
        p[s] = {"frame": f"{e['mission']}_{e['ts_ns']}", "mission": e["mission"],
                "ts_ns": e["ts_ns"], "uv": [round(e["uv"][0], 1), round(e["uv"][1], 1)]}
        if e.get("rhe_index") is not None:
            p[s]["rhe_index"] = e["rhe_index"]
    return p


def as_queued(r):
    """The queue entry whose id is r["id"], or None. The server may swap A/B without rehashing, so both orders are tried."""
    q = from_record(r)
    prefix = r["id"].split(":", 1)[0]
    for cand in (q, {**q, "a": q["b"], "b": q["a"]}):
        if pair_id(prefix, cand) == r["id"]:
            return cand
    return None


def write_relabels(args, rng, carry, records):
    """Re-ask answered pairs with endpoints swapped, in their own queue file and id prefix.

    Not duplicates: these compare protocols, scored by `report.relabels`, not `report.duplicates`.
    """
    # pairs already re-asked are spent (a third ask would measure recall)
    asked = {r[k] for r in records for k in ("dup_of", "relabel_of") if r.get(k)}
    src = [p for p in carry
           if not p["dup_of"] and p["pair_source"] != "hunt" and p["id"] not in asked]
    if not src:
        raise SystemExit(f"no un-re-asked answered pairs in {args.split}")
    n = args.relabel_n or round(args.relabel_frac * len(src))
    n = min(n, len(src))
    pick = [src[i] for i in sorted(rng.choice(len(src), n, replace=False))]
    split = f"{args.split}_relabel"
    out = []
    for q in pick:
        d = {"kind": q["kind"], "pair_source": q["pair_source"], "seg_rel": q.get("seg_rel"),
             "a": q["b"], "b": q["a"], "dup_of": None, "relabel_of": q["id"], "split": split}
        out.append({**d, "id": pair_id(split, d)})
    path = os.path.join(args.out, f"pairs_{split}.jsonl")
    with open(path, "w") as f:
        for d in out:
            f.write(json.dumps(d) + "\n")
    kinds = collections.Counter(d["kind"] for d in out)
    print(f"{len(out)} relabels of {len(src)} not-yet-re-asked answered pairs -> {path}")
    print(f"  {dict(kinds)}; serve with --split {split} --no-regions")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--cfg", default=common.CFG_FILE)
    ap.add_argument("--out", default="side_experiments/wrizz_labeling/output")
    ap.add_argument("--rhe-frac", type=float, default=None,
                    help="share of intra pairs on the driven path; default 0.15, 0 under --anchor-split")
    ap.add_argument("--anchor-split", default=None,
                    help="draw the far endpoint of each cross pair from this split's driven path")
    ap.add_argument("--anchor-frac", type=float, default=1.0,
                    help="anchored cross pairs per frame of this split")
    ap.add_argument("--intra-frac", type=float, default=1.0,
                    help="intra pairs per frame of this split")
    ap.add_argument("--relabel-frac", type=float, default=0.0,
                    help="re-ask this share of answered, not yet re-asked pairs into "
                         "pairs_<split>_relabel.jsonl; generates nothing else")
    ap.add_argument("--relabel-n", type=int, default=0,
                    help="exact number of re-asks instead of a share")
    ap.add_argument("--hunt-frac", type=float, default=0.0,
                    help="share of intra pairs the annotator places (train splits only)")
    ap.add_argument("--cross-mission-frac", type=float, default=0.3)
    ap.add_argument("--same-segment-frac", type=float, default=SAME_SEG_FRAC,
                    help="share of intra pairs inside one segment (annotator check, expect ties)")
    # a share, not a count: 200 duplicates would be over half of near_forest's queue
    ap.add_argument("--duplicate-frac", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true",
                    help="regenerate even if answered pairs would drop out of the queue")
    args = ap.parse_args()

    if args.hunt_frac and args.split not in common.HUNT_SPLITS:
        raise SystemExit(f"hunt pairs bias an eval split; allowed only on {common.HUNT_SPLITS}")

    cfg = common.load_cfg(args.cfg)
    rng = np.random.default_rng(args.seed)
    rhe_frac = args.rhe_frac if args.rhe_frac is not None else (0.0 if args.anchor_split else 0.15)
    all_frames = json.load(open(os.path.join(args.out, "frames.json")))
    frames = [f for f in all_frames if f.get("split") == args.split]
    if not frames:
        raise SystemExit(f"no frames for split {args.split}; run select_frames.py first")
    # span-filtered since frames.json may hold old id_heldout keyframes outside the ID spans
    anchor_frames = [f for f in all_frames if f.get("split") == args.anchor_split
                     and common.frame_mask(args.anchor_split, f["mission"], [f["ts_ns"]])[0]]
    if args.anchor_split and not anchor_frames:
        raise SystemExit(f"no frames for anchor split {args.anchor_split}")

    # Answered pairs are carried over verbatim (regenerating would re-randomize points).
    # All splits' answers count: an id prefix is the split at ask time, not the frames'
    # current split (a fold change stranded 526 of 577 answers). Ids stay as they were.
    records = common.live_records(common.read_jsonl(os.path.join(args.out, "labels.jsonl")))
    have = {f"{f['mission']}_{f['ts_ns']}" for f in frames}
    answered, carry, busy = set(), [], set()
    for r in records:
        q = as_queued(r)
        if q is None:
            continue
        sides = [q[s]["frame"] in have for s in ("a", "b")]
        if not any(sides):
            continue
        answered.add(r["id"])
        # a pair straddling two splits is in neither queue, but its frames are spent
        busy |= {(q[s]["mission"], q[s]["ts_ns"]) for s in ("a", "b")}
        if all(sides):
            carry.append(dict(q, id=r["id"], split=args.split, dup_of=r["dup_of"]))
    frames = [f for f in frames if (f["mission"], f["ts_ns"]) not in busy]
    if answered:
        print(f"carrying {len(carry)} of {len(answered)} answered pairs touching this "
              f"split, on {len(busy)} frames")

    if args.relabel_frac or args.relabel_n:
        write_relabels(args, rng, carry, records)
        return

    geoms = {}

    def geom_of(mission):
        if mission not in geoms:
            geoms[mission] = MissionGeometry(os.path.join(cfg.DATASET.DATA_ROOT, mission),
                                             cfg.DATASET.DEPTH_CACHE_ROOT, cache=4)
        return geoms[mission]

    seg_of = segments(args.out)
    n_rhe = round(rhe_frac * len(frames))
    n_hunt = round(args.hunt_frac * len(frames))
    pairs = rhe_points(cfg, frames, rng, n_rhe, geom_of, seg_of)

    # one intra pair per remaining frame (excluded by frame, not by count)
    used = {(p["a"]["mission"], p["a"]["ts_ns"]) for p in pairs}
    rest = [i for i in rng.permutation(len(frames))
            if (frames[i]["mission"], frames[i]["ts_ns"]) not in used]
    rest = rest[: round(args.intra_frac * len(rest))]
    n_same, no_seg, barren = round(args.same_segment_frac * (len(rest) - n_hunt)), 0, 0
    for k, i in enumerate(rest):
        f = frames[i]
        if k < n_hunt:
            pairs.append({"kind": "intra", "pair_source": "hunt", "seg_rel": None,
                          "a": endpoint(f, (0.0, 0.0)), "b": endpoint(f, (0.0, 0.0))})
            continue
        idmap = seg_of(f["name"])
        no_seg += idmap is None
        want_same = k - n_hunt < n_same
        uv = sample_intra(rng, idmap, want_same)
        if uv:
            pairs.append({"kind": "intra", "pair_source": "random",
                          "seg_rel": "same" if want_same else "diff",
                          "a": endpoint(f, uv[0]), "b": endpoint(f, uv[1])})
        else:
            barren += 1

    # Cross pairs: one per frame against the anchor split's driven path, else one per two
    # frames coupled without replacement. A frame appears at most once intra, once cross.
    n_anchor = 0
    if args.anchor_split:
        want = round(args.anchor_frac * len(frames))
        anchors = anchor_points(cfg, anchor_frames, rng, want, geom_of)
        if len(anchors) < want:
            print(f"  only {len(anchors)} of {want} anchor frames show a path state")
        cross_frames = [frames[i] for i in rng.permutation(len(frames))[: len(anchors)]]
        for f, a in zip(cross_frames, anchors):
            pairs.append({"kind": "cross", "pair_source": "anchor", "seg_rel": None,
                          "a": endpoint(f, sample_point(rng)), "b": a})
        n_anchor = len(cross_frames)
    else:
        for i, j in cross_couples(frames, rng, args.cross_mission_frac):
            pairs.append({"kind": "cross", "pair_source": "random", "seg_rel": None,
                          "a": endpoint(frames[i], sample_point(rng)),
                          "b": endpoint(frames[j], sample_point(rng))})

    # already-answered ground is off limits
    placed = collections.defaultdict(list)
    for r in records:
        for s in ("a", "b"):
            e = r[s]
            if e.get("status") == "ok" and e.get("p_world"):
                placed[e["mission"]].append(e["p_world"])
    pairs, dropped, moved, crowded = repair(pairs, rng, geom_of, seg_of, placed)
    rng.shuffle(pairs)
    for p in pairs:
        p.update(id=pair_id(args.split, p), split=args.split, dup_of=None)

    # duplicates go last (later session), endpoints swapped
    dupes = [pairs[i] for i in rng.choice(len(pairs), min(round(args.duplicate_frac * len(pairs)), len(pairs)),
                                replace=False)
             if pairs[i]["pair_source"] != "hunt"]
    for p in dupes:
        d = {**p, "dup_of": p["id"], "a": p["b"], "b": p["a"]}
        pairs.append({**d, "id": pair_id(args.split, d)})

    # left point becomes A before ids are taken
    for p in pairs:
        if p["a"]["frame"] == p["b"]["frame"] and p["a"]["uv"][0] > p["b"]["uv"][0]:
            p["a"], p["b"] = p["b"], p["a"]
            p["id"] = pair_id(args.split, p)

    pairs = carry + pairs
    ids = [p["id"] for p in pairs]
    assert len(set(ids)) == len(ids), "pair id collision"

    orphaned = answered - set(ids)
    if orphaned and not args.force:
        raise SystemExit(f"{len(orphaned)} answered pairs are not in the new queue and "
                         f"would drop out of the progress count; pass --force to accept")

    path = os.path.join(args.out, f"pairs_{args.split}.jsonl")
    with open(path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    kinds = {k: sum(p["pair_source"] == k for p in pairs) for k in
             ("random", "rhe_overlap", "anchor", "hunt")}
    cross = [p for p in pairs if p["kind"] == "cross"]
    n_cm = sum(p["a"]["mission"] != p["b"]["mission"] for p in cross)
    print(f"{len(pairs)} pairs -> {path}")
    print(f"  {kinds}, cross {len(cross)}, duplicates {len(dupes)}")
    if args.anchor_split:
        print(f"  {n_anchor} cross pairs anchored on the {args.anchor_split} driven path")
    else:
        print(f"  cross-mission {n_cm}/{len(cross)} = {n_cm / max(len(cross), 1):.3f} "
              f"(requested {args.cross_mission_frac})")
    rel = {r: sum(p.get("seg_rel") == r for p in pairs) for r in ("diff", "same")}
    print(f"  intra segment relation {rel}, {barren} frames too uniform to ask, "
          f"{no_seg} frames without a segment map")
    print(f"  anchor repair: {moved} endpoints re-placed, {dropped} pairs dropped, "
          f"{crowded} re-draws rejected for landing within {MIN_WORLD_M} m of another "
          f"point, {len(orphaned)} answered pairs orphaned")


if __name__ == "__main__":
    main()
