"""Queue of answered HDR-eligible eval pairs (an endpoint in a validation mission) for re-asking.

Every annotator gets the same queue, so the sample is drawn once and redrawing after
answers exist is refused.

Run: uv run python side_experiments/wrizz_labeling/schedule_interrater.py
"""

import argparse
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from schedule_pairs import from_record, pair_id

SPLIT = "interrater"


def population(out, cfg):
    """Answered pairs the reported HDR is measured over, as queue entries."""
    valid = set(cfg.DATASET.VALID_MISSIONS)
    frames = {e["name"] for e in json.load(open(os.path.join(out, "frames.json")))}
    seen, pop = set(), []
    for r in common.live_records(common.read_jsonl(os.path.join(out, "labels.jsonl"))):
        if r["t"] is None or r["dup_of"] or not common.hdr_eligible(r):
            continue
        if not {r["a"]["mission"], r["b"]["mission"]} & valid or r["id"] in seen:
            continue
        seen.add(r["id"])
        q = from_record(r)
        if all(q[s]["frame"] in frames for s in ("a", "b")):
            pop.append(dict(q, orig=r["id"]))
    return pop


def sample(pop, n, seed):
    """Sample proportionally over intra/cross kinds."""
    by_kind = collections.defaultdict(list)
    for q in pop:
        by_kind[q["kind"]].append(q)
    rng = np.random.default_rng(seed)
    out, order = [], sorted(by_kind)
    take = {k: round(n * len(by_kind[k]) / len(pop)) for k in order}
    take[order[-1]] = n - sum(take[k] for k in order[:-1])
    for k in order:
        idx = rng.choice(len(by_kind[k]), min(take[k], len(by_kind[k])), replace=False)
        out += [by_kind[k][i] for i in sorted(idx)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="side_experiments/wrizz_labeling/output")
    ap.add_argument("--cfg", default=common.CFG_FILE)
    ap.add_argument("-n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true",
                    help="redraw even if the queue already has answers")
    args = ap.parse_args()

    out = os.path.abspath(args.out)
    path = os.path.join(out, f"pairs_{SPLIT}.jsonl")
    answered = {r["id"] for r in common.live_records(
        common.read_jsonl(os.path.join(out, "labels.jsonl")))}
    if os.path.exists(path) and not args.force:
        done = sum(1 for q in common.read_jsonl(path) if q["id"] in answered)
        if done:
            raise SystemExit(f"{path} already has {done} answers; --force to redraw")

    pop = population(out, common.load_cfg(args.cfg))
    if args.n > len(pop):
        raise SystemExit(f"asked for {args.n} of a population of {len(pop)}")
    pick = sample(pop, args.n, args.seed)

    with open(path, "w") as f:
        for q in pick:
            # endpoints swapped, as for duplicates, to cancel side bias
            d = {"kind": q["kind"], "pair_source": q["pair_source"], "dup_of": None,
                 "relabel_of": q["orig"], "split": SPLIT, "a": q["b"], "b": q["a"]}
            f.write(json.dumps({**d, "id": pair_id(SPLIT, d)}) + "\n")

    kinds = collections.Counter(q["kind"] for q in pick)
    miss = collections.Counter(m for q in pick for m in {q["a"]["mission"], q["b"]["mission"]})
    print(f"{len(pick)} of {len(pop)} eval pairs -> {path}")
    print(f"  {dict(kinds)}; missions {dict(miss.most_common())}")
    print(f"  serve with --split {SPLIT} --no-regions --no-reveal --annotator <id>")


if __name__ == "__main__":
    main()
