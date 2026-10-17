"""Pick keyframes for a split by DINO feature diversity (greedy k-center, equal covering radius per mission).

Run: uv run python side_experiments/wrizz_labeling/select_frames.py --split id_heldout"""

import argparse
import collections
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common

# Total keyframes per split. Marginal coverage per keyframe halves between k=100 and 250.
# Tiers: 120 keyframes buy their 180 pairs (one anchored cross per keyframe, one intra per two).
# Flat across tiers so each quadrant of figures/ood/quadrant.py gets similar precision.
TIER_BUDGETS = {"near_roots": 120, "near_undergrowth": 120, "near_meadow": 120,
                "medium_alpine": 120, "far_construction_outdoor": 120, "far_snow": 120,
                "far_urban": 120, "far_indoor": 120}
BUDGETS = {"id_heldout": 480, "near_forest": 240, "id_train": 1200, **TIER_BUDGETS}
# Cap on a site's multiple of its uniform share. id_heldout is looser: at 1.3 RIV-1 and
# HIL-1 both pinned to the cap (RIV-1 spread 0.078 vs 0.021 for ALB-3).
MAX_SHARE = {"id_heldout": 1.8}
DEFAULT_MAX_SHARE = 1.3
# Missions open and close standing still or facing the operator
TRIM = 0.1
KMAX = 400
# Upper bound on the per-site floor
KMIN = 20


def mission_features(cfg, mission, split):
    """(ts_ns, pooled features) with head/tail trimmed and out-of-span or excluded frames cut.

    Cut before k-center so they can neither be picked nor shift the centroid.
    """
    ts_ns, X = common.pooled_features(os.path.join(cfg.DATASET.DINO_CACHE_ROOT, mission))
    lo, hi = int(TRIM * len(ts_ns)), int((1 - TRIM) * len(ts_ns))
    ts_ns, X = ts_ns[lo:hi], X[lo:hi]
    keep = common.frame_mask(split, mission, ts_ns)
    if not keep.all():
        print(f"  {mission}: {len(ts_ns) - int(keep.sum())} of {len(ts_ns)} frames cut, "
              f"outside {split}", flush=True)
    return ts_ns[keep], X[keep]


def group_curve(feats, missions, protected):
    """((mission, ts_ns) of the greedy picks, covering radius before each pick).

    One k-center run over the whole group, seeded with already-labeled frames.
    """
    names, stamps = [], []
    for m in missions:
        ts_ns, _ = feats[m]
        names += [m] * len(ts_ns)
        stamps.append(ts_ns)
    stamps = np.concatenate(stamps)
    X = np.concatenate([feats[m][1] for m in missions])
    seeds = [i for i, (m, t) in enumerate(zip(names, stamps)) if f"{m}_{t}" in protected]
    picks, radii = common.k_center(X, KMAX, seeds, ts=stamps)
    return [(names[i], int(stamps[i])) for i in picks], radii


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--cfg", default=common.CFG_FILE)
    ap.add_argument("--out", default="side_experiments/wrizz_labeling/output")
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--max-share", type=float, default=None)
    ap.add_argument("--kmin", type=int, default=None,
                    help="floor on a site's keyframes; defaults to the uniform share")
    args = ap.parse_args()

    cfg = common.load_cfg(args.cfg)
    missions = common.split_missions(cfg, args.split)
    budget = args.budget or BUDGETS.get(args.split)
    if budget is None:
        raise SystemExit(f"no default budget for {args.split}, pass --budget")

    frames_dir = os.path.join(args.out, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    manifest_path = os.path.join(args.out, "frames.json")
    entries = {e["name"]: e for e in json.load(open(manifest_path))} if os.path.exists(manifest_path) else {}

    # frames with existing labels are never dropped
    protected = set()
    for r in common.read_jsonl(os.path.join(args.out, "labels.jsonl")):
        for side in ("a", "b"):
            e = r.get(side)
            if isinstance(e, dict) and e.get("mission"):
                protected.add(f"{e['mission']}_{e['ts_ns']}")

    queued = set()
    for name in os.listdir(args.out):
        if name.startswith("pairs_") and name.endswith(".jsonl"):
            for q in common.read_jsonl(os.path.join(args.out, name)):
                queued |= {q["a"]["frame"], q["b"]["frame"]}

    max_share = args.max_share or MAX_SHARE.get(args.split, DEFAULT_MAX_SHARE)
    print(f"loading features for {len(missions)} missions", flush=True)
    feats = {m: mission_features(cfg, m, args.split) for m in missions}
    sites = common.redundant_groups({m: X for m, (_, X) in feats.items()})
    print(f"{args.split}: {len(sites)} groups (budget {budget}, max share {max_share})",
          flush=True)
    curves, order = {}, {}
    for site, ms in sites.items():
        order[site], curves[site] = group_curve(feats, ms, protected)
        print(f"  {site}: {len(order[site])} candidates", flush=True)
    kmin = args.kmin or max(4, min(KMIN, budget // len(sites)))
    counts, radius = common.allocate(curves, budget, kmin=kmin, max_share=max_share)

    # labeled frames are kept outside the budget (they seeded k-center), but only inside this
    # split's spans, since a mission can feed two tiers
    def in_span(name):
        m, t = name.rsplit("_", 1)
        return m in set(missions) and common.span_mask(args.split, m, [int(t)])[0]

    keep = {n for n in protected if in_span(n)}
    for site in sites:
        keep |= {f"{m}_{t}" for m, t in order[site][: counts[site]]}
    # excluded stretches override protection; export drops their labels
    keep = {n for n in keep
            if not common.excluded(n.rsplit("_", 1)[0], int(n.rsplit("_", 1)[1]))}
    elsewhere = [n for n in protected
                 if n.rsplit("_", 1)[0] in set(missions) and not in_span(n)]
    if elsewhere:
        print(f"  NOTE: {len(elsewhere)} already-labeled frames of these missions sit "
              f"outside the {args.split} spans and stay with their current split")
    per_mission = collections.Counter(n.rsplit("_", 1)[0] for n in keep)
    print(f"covering radius {radius:.4f} -> " +
          "  ".join(f"{s}:{counts[s]}" for s in sites))
    print("  per mission " + "  ".join(f"{m}:{per_mission[m]}" for m in missions))

    # drop this split's unlabeled surplus, keep every other split untouched
    dropped = [n for n, e in entries.items()
               if e.get("split") == args.split and n not in keep
               and (n not in protected or common.excluded(e["mission"], e["ts_ns"]))]
    for n in dropped:
        del entries[n]

    added, moved = 0, []
    want = {}
    for name in keep:
        m, t = name.rsplit("_", 1)
        want.setdefault(m, []).append(int(t))
    for m in missions:
        if m not in want:
            continue
        m_ts, m_paths = common.mission_frames(cfg, m, native=True)
        for t in want[m]:
            name = f"{m}_{t}"
            if name in entries:
                # same photo, retiered: only its split changes
                if entries[name]["split"] != args.split:
                    entries[name]["split"] = args.split
                    moved.append(name)
                continue
            j = int(np.searchsorted(m_ts, t))
            assert m_ts[j] == t, f"{name}: dino cache ts not in image manifest"
            img = cv2.imread(m_paths[j], -1)
            assert img is not None, m_paths[j]
            # 2x the coordinate grid (intra panel size)
            size = (common.SIZE[0] * 2, common.SIZE[1] * 2)
            if (img.shape[1], img.shape[0]) != size:
                img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(frames_dir, name + ".jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 94])
            entries[name] = {"name": name, "mission": m, "ts_ns": t,
                             "split": args.split, "w": common.SIZE[0], "h": common.SIZE[1]}
            added += 1

    ordered = sorted(entries.values(), key=lambda e: (e["split"], e["mission"], e["ts_ns"]))
    json.dump(ordered, open(manifest_path, "w"), indent=1)
    n_split = sum(1 for e in ordered if e["split"] == args.split)
    print(f"{args.split}: {n_split} frames (+{added} added, -{len(dropped)} dropped, "
          f"{len(moved)} retagged from another split), {len(ordered)} total in {manifest_path}")
    orphaned = queued & set(dropped)
    if orphaned:
        print(f"  WARNING: {len(orphaned)} dropped frames are still referenced by a "
              f"queue; re-run schedule_pairs.py --split {args.split} before labeling")


if __name__ == "__main__":
    main()
