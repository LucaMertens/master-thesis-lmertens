"""Human vs RHE agreement, label consistency stats, and a crosshair sheet (blue = more traversable).

Run: uv run python side_experiments/wrizz_labeling/report.py --sheet out.png"""

import argparse
import collections
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
import rhe as rhe_mod
from export_labels import merge_amendments
from traversability_network.adaptations import propagate as P
from traversability_network.adaptations.visualize_rizz_pairs import EQUAL, LESS, MORE, crosshair


def resolve(labels, cfg):
    """RHE (mu, obs) per endpoint of every answered pair, recomputed from the click as export does."""
    rhe_cache, geom_cache, out = {}, {}, []
    for r in labels:
        if r["t"] is None:
            continue
        mus = []
        for side in ("a", "b"):
            e = r[side]
            m = e["mission"]
            if m not in rhe_cache:
                d = os.path.join(cfg.DATASET.DATA_ROOT, m)
                rhe_cache[m] = rhe_mod.RheLabels(cfg.DATASET.LABELS_ROOT, m, d)
                geom_cache[m] = P.MissionGeometry(d, cfg.DATASET.DEPTH_CACHE_ROOT, cache=4)
            labels_m, geom = rhe_cache[m], geom_cache[m]
            if e.get("rhe_index") is not None:
                i = e["rhe_index"]
                mus.append((float(labels_m.mu[i]), float(labels_m.obs[i])))
                continue
            a = P.anchor_from_pixel(e["uv"], geom.frame_index(e["ts_ns"]), geom, mode="auto")
            hit = labels_m.at(a["p_world"]) if a["status"] == "ok" else None
            mus.append((hit["mu"], hit["obs"]) if hit else None)
        out.append((r, mus))
    return out


def agreement(resolved, verbose=True):
    """Human vs RHE disagreement. The headline uses common.EVAL_SOURCES only; biased sources are separate."""
    both = [(r, m) for r, m in resolved if m[0] and m[1]]
    out = {"answered": len(resolved), "covered": len(both)}
    # per split, so ID splits can be read alone
    cov = collections.Counter(r["split"] for r, _ in both)
    out["by_split"] = {sp: {"answered": n, "covered": cov[sp]} for sp, n in
                       collections.Counter(r["split"] for r, _ in resolved).items()}
    rows = []
    for r, (a, b) in both:
        t_rhe = rhe_mod.compare(a[0], b[0])
        if t_rhe is not None:
            rows.append((r["t"] == t_rhe, min(a[1], b[1]), abs(b[0] - a[0]),
                         r["pair_source"], common.hdr_eligible(r), r["t"], t_rhe,
                         r["split"]))
    out["biased"] = {src: (n, float(1 - np.mean([x[0] for x in rows
                                                 if x[3] == src and not x[4]])))
                     for src, n in collections.Counter(
                         x[3] for x in rows if not x[4]).items()}
    rows = [x for x in rows if x[4]]
    out["n"] = len(rows)
    if rows:
        ok = np.array([x[0] for x in rows])
        gap = np.array([x[2] for x in rows])
        t_h = np.array([x[5] for x in rows])
        t_r = np.array([x[6] for x in rows])
        out["disagreement"] = float(1 - ok.mean())
        # tie vs ordering is a boundary call; opposite orderings are a reversal
        reversal = (t_h == -t_r) & (t_h != 0)
        out["agree_n"] = int(ok.sum())
        out["reversal_n"] = int(reversal.sum())
        out["boundary_n"] = int((~ok & ~reversal).sum())
        # RHE calls most pairs a tie, so the pooled number mostly reflects the RHE deadband
        tie = t_r == 0
        out["rhe_tie_frac"] = float(tie.mean())
        out["tie_n"] = int(tie.sum())
        out["tie_disagreement"] = float(1 - ok[tie].mean()) if tie.any() else None
        out["ineq_n"] = int((~tie).sum())
        out["ineq_disagreement"] = float(1 - ok[~tie].mean()) if (~tie).any() else None
        out["human_tie_on_rhe_tie"] = float(np.mean(t_h[tie] == 0)) if tie.any() else None
        for sp in out["by_split"]:
            m = np.array([x[7] == sp for x in rows])
            out["by_split"][sp].update(
                n=int(m.sum()), agree_n=int((ok & m).sum()),
                reversal_n=int((reversal & m).sum()),
                boundary_n=int((~ok & ~reversal & m).sum()),
                rhe_tie_n=int((tie & m).sum()))
        out["by_gap"] = []
        for lo, hi in ((0.0, 0.125), (0.125, 0.25), (0.25, 0.5), (0.5, 1.01)):
            m = (gap >= lo) & (gap < hi)
            if m.sum():
                out["by_gap"].append((lo, hi, int(m.sum()), float(1 - ok[m].mean())))
    if verbose:
        print(f"answered {out['answered']}, both endpoints RHE-covered {out['covered']} "
              f"({100 * out['covered'] / max(1, out['answered']):.0f}%)")
        for src, (n, dis) in sorted(out["biased"].items()):
            print(f"  [biased sample, not the headline] {src}: n={n}, disagreement {dis:.3f}")
        if not rows:
            print(f"  no {'/'.join(common.EVAL_SOURCES)} pairs are RHE-comparable yet")
        else:
            print(f"comparable {'/'.join(common.EVAL_SOURCES)} pairs {out['n']}, "
                  f"disagreement {out['disagreement']:.3f}")
            print(f"  RHE calls {out['rhe_tie_frac']:.0%} of them a tie; "
                  f"disagreement there {out['tie_disagreement']:.3f} "
                  f"(human agrees it is a tie {out['human_tie_on_rhe_tie']:.0%} of the time)")
            if out["ineq_n"]:
                print(f"  RHE inequality pairs n={out['ineq_n']}, "
                      f"disagreement {out['ineq_disagreement']:.3f}")
            print("  by RHE mu gap:")
            for lo, hi, n, dis in out["by_gap"]:
                print(f"    {lo:.3f}-{hi:.3f}  n={n:4d}  disagreement {dis:.3f}")
    return out


def _endpoints(r):
    return tuple((e["mission"], e["ts_ns"], tuple(e["uv"])) for e in (r["a"], r["b"]))


def orientation(a, b):
    """-1 if the re-ask has the endpoints swapped, +1 if in order, 0 if not the same pair.

    Read from the endpoints: the server re-normalizes intra pairs, which swaps a third back.
    """
    ends, re_ends = _endpoints(a), _endpoints(b)
    if ends == re_ends[::-1]:
        return -1
    return 1 if ends == re_ends else 0


def _consistency(pairs):
    """Orientation-corrected agreement over (original, re-ask) pairs, with reversals split out."""
    pairs = [(a, b, s) for a, b in pairs if (s := orientation(a, b))]
    t_a = np.array([a["t"] for a, _, _ in pairs])
    t_b = np.array([s * b["t"] for _, b, s in pairs])
    agree = t_a == t_b
    reversal = (t_a == -t_b) & (t_a != 0)
    return {"n": len(pairs), "agree_n": int(agree.sum()),
            "boundary_n": int((~agree & ~reversal).sum()), "reversal_n": int(reversal.sum()),
            "disagreement": float(1 - agree.mean()) if len(pairs) else None,
            "reversal": float(reversal.mean()) if len(pairs) else None}


def duplicates(labels, verbose=True):
    """Human self-disagreement: the empirical ceiling on HDR."""
    by_id = {r["id"]: r for r in labels}
    pairs = [(by_id[r["dup_of"]], r) for r in labels
             if r.get("dup_of") and r["dup_of"] in by_id and r["t"] is not None]
    pairs = [(a, b) for a, b in pairs if a["t"] is not None]
    cross = [(a, b) for a, b in pairs if a.get("session") != b.get("session")]
    c = _consistency(cross)
    out = {"n": len(pairs), "n_cross_session": c.pop("n"), **c}
    out["self_disagreement"] = out.pop("disagreement")
    if verbose:
        if not pairs:
            print("duplicates: none answered yet")
        else:
            print(f"duplicates {len(pairs)}, {len(cross)} in a different session")
            if cross:
                print(f"  self-disagreement {out['self_disagreement']:.3f}, of which "
                      f"{out['reversal_n']} are a reversal ({out['reversal']:.3f})")
            if len(pairs) - len(cross):
                print(f"  WARNING: {len(pairs) - len(cross)} answered in the same session "
                      f"as their original, so they measure short-term recall")
    return out


def reasks(labels, verbose=True):
    """Cross-session re-asks pooled over `duplicates` and `relabels`."""
    by_id = {r["id"]: r for r in labels}
    pairs = [(by_id[src], r) for r in labels for k in ("dup_of", "relabel_of")
             if (src := r.get(k)) and src in by_id
             and r["t"] is not None and by_id[src]["t"] is not None]
    cross = [(a, b) for a, b in pairs if a.get("session") != b.get("session")]
    out = _consistency(cross)
    out["n_all"] = len(pairs)
    # region_m unset means the whole segment, so only an explicit 0 is point-only
    def regioned(r):
        return any(r[s].get("region_m") != 0 for s in ("a", "b"))

    out["n_same_protocol"] = sum(1 for a, b in cross if regioned(a) == regioned(b))
    if verbose:
        print(f"re-asks {out['n_all']}, {out['n']} in a different session")
        if out["n"]:
            print(f"  disagreement {out['disagreement']:.3f}, of which "
                  f"{out['reversal_n']} are a reversal ({out['reversal']:.3f})")
            print(f"  {out['n_same_protocol']} under the original's region protocol")
    return out


def relabels(labels, verbose=True):
    """Same pair answered under two protocols: does dropping the SAM region change the answer?"""
    by_id = {r["id"]: r for r in labels}
    pairs = [(by_id[r["relabel_of"]], r) for r in labels
             if r.get("relabel_of") and r["relabel_of"] in by_id and r["t"] is not None]
    pairs = [(a, b) for a, b in pairs if a["t"] is not None]
    out = _consistency(pairs)
    out["sign_flips"] = out["reversal_n"]
    regioned = [(a, b) for a, b in pairs
                if max(a["a"].get("region_m") or 0, a["b"].get("region_m") or 0) > 0]
    out["disagreement_regioned"] = _consistency(regioned)["disagreement"]
    if verbose:
        if not pairs:
            print("relabels: none answered yet")
        else:
            print(f"relabels {out['n']}")
            print(f"  disagreement with the original {out['disagreement']:.3f}, "
                  f"{out['sign_flips']} sign flips")
            if out["disagreement_regioned"] is not None:
                print(f"  where the original carried a region: "
                      f"{out['disagreement_regioned']:.3f} over {len(regioned)}")
    return out


def check_pairs(labels, verbose=True):
    """Planted same-segment pairs: expected answer is a tie, so non-ties are attention slips."""
    chk = [r for r in labels if r.get("seg_rel") == "same" and r["t"] is not None]
    out = {"n": len(chk),
           "tie_rate": float(np.mean([r["t"] == 0 for r in chk])) if chk else None}
    if verbose and chk:
        print(f"check pairs {out['n']}, tie rate {out['tie_rate']:.3f}")
    return out


def timing(labels, verbose=True):
    """Wall time actually spent answering, per split."""
    per = {}
    for r in labels:
        dt = r.get("dt_ms")
        if dt:
            per.setdefault(r["split"], []).append(dt / 1000)
    out = {k: {"n": len(v), "hours": float(np.sum(v)) / 3600,
               "median_s": float(np.median(v))} for k, v in per.items()}
    if verbose:
        for sp, d in sorted(out.items()):
            print(f"{sp:11s} n={d['n']:5d}  {d['hours']:5.2f} h  median {d['median_s']:.1f} s")
    return out


def sheet(labels, out_dir, path, n=40, cols=4):
    tiles = []
    for r in labels[:n]:
        if r["t"] is None:
            continue
        panels = []
        for side, other in (("a", "b"), ("b", "a")):
            e = r[side]
            img = cv2.imread(os.path.join(out_dir, "frames", f"{e['mission']}_{e['ts_ns']}.jpg"))
            if img is None:
                # a one-panel cross tile would look like an intra pair
                panels = []
                break
            uv = common.display_pixel(e["uv"])
            t = r["t"]
            color = EQUAL if t == 0 else ((MORE if side == "a" else LESS) if t < 0
                                          else (LESS if side == "a" else MORE))
            crosshair(img, uv, color, side.upper())
            panels.append(img)
            if r["a"]["ts_ns"] == r["b"]["ts_ns"] and r["a"]["mission"] == r["b"]["mission"]:
                # intra pair: both crosshairs on one image
                e2 = r[other]
                uv2 = common.display_pixel(e2["uv"])
                c2 = EQUAL if t == 0 else ((LESS if side == "a" else MORE) if t < 0
                                           else (MORE if side == "a" else LESS))
                crosshair(img, uv2, c2, other.upper())
                break
        if panels:
            tiles.append(np.hstack(panels) if len(panels) == 2 else panels[0])
    if not tiles:
        print("sheet: nothing to draw")
        return
    # intra tiles hold one image, cross tiles two; scale rather than pad
    w = max(t.shape[1] for t in tiles)
    h = max(t.shape[0] for t in tiles)
    tiles = [t if t.shape[:2] == (h, w) else cv2.resize(t, (w, h), interpolation=cv2.INTER_AREA)
             for t in tiles]
    rows = [np.hstack(tiles[i:i + cols] + [np.zeros_like(tiles[0])] * (cols - len(tiles[i:i + cols])))
            for i in range(0, len(tiles), cols)]
    cv2.imwrite(path, np.vstack(rows))
    print(f"sheet: {len(tiles)} pairs -> {path}  (blue = more traversable, red = less, yellow = equal)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="side_experiments/wrizz_labeling/output")
    ap.add_argument("--cfg", default=common.CFG_FILE)
    ap.add_argument("--sheet", default=None)
    ap.add_argument("--sheet-n", type=int, default=40)
    args = ap.parse_args()

    cfg = common.load_cfg(args.cfg)
    labels = merge_amendments(common.read_jsonl(os.path.join(args.out, "labels.jsonl")))
    if not labels:
        raise SystemExit("no labels yet")

    counts = collections.Counter(r["pair_source"] for r in labels)
    skipped = sum(r["t"] is None for r in labels)
    print(f"{len(labels)} labels {dict(counts)}, skipped {skipped}")
    agreement(resolve(labels, cfg))
    duplicates(labels)
    relabels(labels)
    check_pairs(labels)
    timing(labels)
    if args.sheet:
        sheet(labels, args.out, args.sheet, args.sheet_n)


if __name__ == "__main__":
    main()
