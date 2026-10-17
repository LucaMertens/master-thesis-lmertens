"""Backend for the pairwise labeling UI. Blind then reveal: RHE is only returned after the answer is saved.

Run: uv run python side_experiments/wrizz_labeling/server.py"""

import argparse
import json
import os
import collections
import statistics
import sys
import time
import threading
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
import rhe as rhe_mod
from traversability_network.adaptations.propagate import (
    REGION_FULL_M, MissionGeometry, anchor_from_pixel, planar_mask, propagate_point,
)

HERE = os.path.dirname(os.path.abspath(__file__))
PREVIEW = 2  # frames each side probed live; the full window runs offline at export
PREFETCH = 12  # pairs encoded ahead of the served batch when --sam is on
# Eval sets first, OOD tiers in tiers.json near-to-far order
SPLIT_ORDER = ["id_heldout", *common.TIER_MISSIONS, "near_forest", "id_train"]
# Fields shown to the annotator. pair_source is withheld: rhe_overlap would reveal both points were driven over.
WIRE_KEYS = ("id", "split", "kind")


def normalize(p):
    """Put the left-hand point of an intra pair in A. The pair id is not recomputed."""
    if p["a"]["frame"] == p["b"]["frame"] and p["a"]["uv"][0] > p["b"]["uv"][0]:
        p["a"], p["b"] = p["b"], p["a"]
    return p


def load_queues(out):
    """Every per-split queue, eval splits first, duplicates last within each split."""
    queues = {}
    for name in os.listdir(out):
        if name.startswith("pairs_") and name.endswith(".jsonl"):
            queues[name[len("pairs_"):-len(".jsonl")]] = common.read_jsonl(os.path.join(out, name))
    rank = {s: i for i, s in enumerate(SPLIT_ORDER)}
    order = sorted(queues, key=lambda s: (rank.get(s, len(rank)), s))
    return [normalize(p) for split in order for p in queues[split]], order


class Store:
    """Queue, answers and the per-mission geometry, loaded lazily."""

    def __init__(self, out, cfg_file, splits=None, skip_dup=False, skip_sources=(),
                 annotator="lm"):
        self.out = out
        # log reads are scoped to this annotator, so several can share one queue
        self.annotator = annotator
        self.cfg = common.load_cfg(cfg_file)
        self.frames = {e["name"]: e for e in json.load(open(os.path.join(out, "frames.json")))}
        self.pairs, self.split_order = load_queues(out)
        # session filters; queue files are untouched
        if splits:
            self.pairs = [p for p in self.pairs if p["split"] in splits]
        if skip_dup:
            self.pairs = [p for p in self.pairs if not p["dup_of"]]
        if skip_sources:
            self.pairs = [p for p in self.pairs if p["pair_source"] not in skip_sources]
        if not self.pairs:
            raise SystemExit("no pairs left to serve after --split/--skip filters")
        # progress() indexes per-split counts, so drop filtered splits
        served = {p["split"] for p in self.pairs}
        self.split_order = [x for x in self.split_order if x in served]
        # fail at startup, not mid-session, on pairs whose photo is gone
        broken = {p["id"] for p in self.pairs
                  for s in ("a", "b") if p[s]["frame"] not in self.frames}
        if broken:
            raise SystemExit(f"{len(broken)} queued pairs reference frames that are not "
                             f"in frames.json; re-run schedule_pairs.py")
        self.labels_path = os.path.join(out, "labels.jsonl")
        self._geom, self._rhe = {}, {}
        self._sam, self.sam_device = None, None
        self.no_regions = False
        self.no_reveal = False
        self._idmaps = {}
        self._warm_q, self._warm_seen = collections.deque(), set()
        # guards every read-modify-append on the log
        self.lock = threading.Lock()
        self.session = datetime.now(timezone.utc).strftime("s%Y%m%d%H%M")
        self.reload()

    def reload(self):
        live = self.own_records()
        self.answered = {r["id"] for r in live}
        self.times = [r["dt_ms"] for r in live if r.get("dt_ms")]

    def own_records(self):
        """Committed answers by this annotator, oldest first."""
        return [r for r in common.live_records(common.read_jsonl(self.labels_path))
                if r.get("annotator", "lm") == self.annotator]

    def geom(self, mission):
        if mission not in self._geom:
            d = os.path.join(self.cfg.DATASET.DATA_ROOT, mission)
            self._geom[mission] = MissionGeometry(d, self.cfg.DATASET.DEPTH_CACHE_ROOT, cache=96)
        return self._geom[mission]

    def rhe(self, mission):
        if mission not in self._rhe:
            d = os.path.join(self.cfg.DATASET.DATA_ROOT, mission)
            self._rhe[mission] = rhe_mod.RheLabels(self.cfg.DATASET.LABELS_ROOT, mission, d)
        return self._rhe[mission]

    def warm(self):
        """Build per-mission geometry up front to avoid a multi-second stall on the first /api/next."""
        missions = sorted({p[s]["mission"] for p in self.pairs
                           if p["id"] not in self.answered for s in ("a", "b")})
        for m in missions:
            self.geom(m)
            print(f"  warmed {m}", flush=True)

    def sam(self):
        """Promptable SAM, built on first use."""
        if self._sam is None:
            import sam as sam_mod
            self._sam = sam_mod.Prompter(os.path.join(self.out, "frames"), self.sam_device)
        return self._sam

    def prefetch(self, names):
        """Queue frames for the encoder to run ahead of the annotator."""
        for n in names:
            if n not in self._warm_seen:
                self._warm_seen.add(n)
                self._warm_q.append(n)

    def _warm_loop(self):
        while True:
            try:
                name = self._warm_q.popleft()
            except IndexError:
                time.sleep(0.4)
                continue
            try:
                self.sam().warm(name)
            except Exception as exc:
                print(f"  prefetch {name}: {exc}", flush=True)

    def start_prefetch(self):
        threading.Thread(target=self._warm_loop, daemon=True).start()

    def segment(self, name, points, labels, prev=None):
        import sam as sam_mod
        m, logits, iou = self.sam().mask(name, points, labels, common.SIZE, prev=prev)
        if iou < sam_mod.MIN_IOU or not m.any():
            return {"ok": False, "iou": iou}
        return {"ok": True, "iou": iou, "rle": sam_mod.rle(m),
                "area": float(m.mean())}

    def frame_of(self, name):
        """(mission, ts_ns) for a frame name. Names, not ts_ns: ts_ns exceeds JS's safe integer range."""
        e = self.frames.get(name)
        if e is None:
            raise KeyError(name)
        return e["mission"], int(e["ts_ns"])

    def path_pixels(self, name):
        """Projected driven path for one frame. Geometry only, no mu."""
        mission, ts_ns = self.frame_of(name)
        geom = self.geom(mission)
        return self.rhe(mission).path_pixels(geom, geom.frame_index(ts_ns))

    def idmap(self, name):
        if name not in self._idmaps:
            self._idmaps[name] = common.load_idmap(
                os.path.join(self.out, "segments", name + ".png"))
            if len(self._idmaps) > 64:
                self._idmaps.pop(next(iter(self._idmaps)))
        return self._idmaps[name]

    def region(self, name, uv, anchor, geom, frame):
        """The segment under a click, cut at the surface break (as export does)."""
        idmap = self.idmap(name)
        if idmap is None or anchor.get("status") != "ok":
            return None
        seg = common.segment_at(idmap, uv)
        if not seg:
            return None
        cut = planar_mask(idmap, seg, anchor, frame, geom, common.SCALE)
        return common.rle(cut) if cut is not None and cut.any() else None

    def anchor(self, mission, ts_ns, uv):
        """Geometry only, no RHE, so safe before an answer."""
        geom = self.geom(mission)
        frame = geom.frame_index(ts_ns)
        a = anchor_from_pixel(uv, frame, geom, mode="auto")
        if a["status"] == "ok":
            f, _, _, _ = propagate_point(a["p_world"], frame, geom, window=PREVIEW, min_side=1)
            a["preview"] = len(f)
            # display px per metre, for the browser's radius clip
            a["px_per_m"] = (geom.K[0, 0] / common.SCALE) / a["z"]
        return a

    def rhe_at(self, endpoint, anchor):
        """RHE mu for an endpoint: the scheduled state if any, else the nearest path state."""
        labels = self.rhe(endpoint["mission"])
        if endpoint.get("rhe_index") is not None:
            i = endpoint["rhe_index"]
            return {"index": i, "mu": float(labels.mu[i]), "obs": float(labels.obs[i]), "dist": 0.0}
        if anchor.get("status") != "ok":
            return None
        return labels.at(anchor["p_world"])

    def next_pairs(self, n):
        out = []
        for p in self.pairs:
            if len(out) >= n:
                break
            if p["id"] in self.answered:
                continue
            hunt = p["pair_source"] == "hunt"
            item = {k: p[k] for k in WIRE_KEYS}
            item["hunt"] = hunt
            for side in ("a", "b"):
                e = p[side]
                # rhe_index is withheld for the same reason as pair_source
                item[side] = {k: v for k, v in e.items() if k != "rhe_index"}
                if not hunt:
                    a = self.anchor(e["mission"], e["ts_ns"], e["uv"])
                    item[side]["anchor"] = a
                    geom = self.geom(e["mission"])
                    item[side]["region_rle"] = None if self.no_regions else self.region(
                        e["frame"], e["uv"], a, geom, geom.frame_index(e["ts_ns"]))
            out.append(item)
        if self.sam_device:
            # served pairs plus a lookahead
            ahead = [q for q in self.pairs if q["id"] not in self.answered][: n + PREFETCH]
            self.prefetch([q[s]["frame"] for q in ahead for s in ("a", "b")])
        return out

    def progress(self):
        by_split = {}
        for p in self.pairs:
            s = by_split.setdefault(p["split"], {"total": 0, "done": 0, "dup_total": 0, "dup_done": 0})
            key = "dup_" if p["dup_of"] else ""
            s[key + "total"] += 1
            s[key + "done"] += p["id"] in self.answered
        recent = self.times[-50:]
        median_ms = statistics.median(recent) if recent else None
        # active split, base pairs only (duplicates and later splits are later sessions)
        active = next((s for s in self.split_order
                       if by_split[s]["done"] < by_split[s]["total"]), None)
        remaining = by_split[active]["total"] - by_split[active]["done"] if active else 0
        return {"splits": by_split, "median_ms": median_ms, "remaining": remaining,
                "active": active, "eta_s": (median_ms / 1000 * remaining) if median_ms else None,
                "session": self.session}


class Handler(BaseHTTPRequestHandler):
    store = None

    def _send(self, code, body=b"", ctype="application/octet-stream"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj).encode(), "application/json")

    def _file(self, path, ctype):
        if not os.path.exists(path):
            return self._send(404, b"not found", "text/plain")
        with open(path, "rb") as f:
            self._send(200, f.read(), ctype)

    def _body(self):
        return json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")

    def do_GET(self):
        p, _, query = self.path.partition("?")
        s = self.store
        if p == "/":
            return self._file(os.path.join(HERE, "static", "index.html"), "text/html")
        if p == "/api/next":
            n = int(dict(q.split("=") for q in query.split("&") if "=" in q).get("n", 3))
            return self._json({"pairs": s.next_pairs(n), "progress": s.progress(),
                               "scale": common.SCALE, "size": list(common.SIZE),
                               "region_full_m": REGION_FULL_M,
                               "no_regions": s.no_regions})
        if p == "/api/path":
            if s.no_reveal:
                return self._send(403, b"path peek disabled", "text/plain")
            a = dict(kv.split("=", 1) for kv in query.split("&") if "=" in kv)
            name = unquote(a.get("frame", ""))
            if name not in s.frames:
                return self._send(404, b"unknown frame", "text/plain")
            return self._json({"path": s.path_pixels(name)})
        if p == "/api/progress":
            return self._json(s.progress())
        for prefix, sub, ext, ctype in (("/frames/", "frames", ".jpg", "image/jpeg"),
                                        ("/segments/", "segments", ".png", "image/png")):
            if p.startswith(prefix):
                name = os.path.splitext(os.path.basename(p))[0]
                if name not in s.frames:
                    return self._send(404, b"unknown frame", "text/plain")
                return self._file(os.path.join(s.out, sub, name + ext), ctype)
        self._send(404, b"not found", "text/plain")

    def do_POST(self):
        p = self.path.split("?")[0]
        s = self.store
        if p == "/api/segment":
            b = self._body()
            if self.store.sam_device is None:
                return self._json({"ok": False, "error": "server started without --sam"}, 503)
            if b["frame"] not in s.frames:
                return self._send(404, b"unknown frame", "text/plain")
            return self._json(s.segment(b["frame"], b["points"], b["labels"]))
        if p == "/api/anchor":
            b = self._body()
            mission, ts_ns = s.frame_of(b["frame"])
            return self._json(s.anchor(mission, ts_ns, b["uv"]))
        if p == "/api/label":
            return self._label(self._body())
        if p == "/api/amend":
            b = self._body()
            with s.lock:
                common.append_jsonl(s.labels_path, {"v": 1, "amend": True, **b,
                                                    "ts": datetime.now(timezone.utc).isoformat()})
            return self._json({"ok": True})
        if p == "/api/undo":
            return self._undo()
        self._send(404, b"not found", "text/plain")

    def _label(self, b):
        """Persist the answer, then compute the RHE reveal."""
        s = self.store
        pair = next((q for q in s.pairs if q["id"] == b["id"]), None)
        if pair is None:
            return self._json({"error": "unknown pair"}, 404)

        record = {"v": 1, "id": pair["id"], "ts": datetime.now(timezone.utc).isoformat(),
                  "annotator": s.annotator, "session": s.session,
                  "split": pair["split"], "kind": pair["kind"],
                  "pair_source": pair["pair_source"], "seg_rel": pair.get("seg_rel"),
                  "dup_of": pair["dup_of"], "relabel_of": pair.get("relabel_of"),
                  "t": b["t"], "confidence": b.get("confidence", 2),
                  "dt_ms": b.get("dt_ms"), "flags": b.get("flags", []),
                  "uuid": uuid.uuid4().hex}
        for side in ("a", "b"):
            e, sent = pair[side], b.get(side, {})
            uv = sent.get("uv", e["uv"])
            anchor = s.anchor(e["mission"], e["ts_ns"], uv) if b["t"] is not None else {}
            extra = {}
            if sent.get("region_rle"):
                # store the mask so export needs no GPU or SAM checkpoint
                extra = {"prompts": sent.get("prompts", []), "region_rle": sent["region_rle"]}
            record[side] = {**extra, "mission": e["mission"], "ts_ns": e["ts_ns"],
                            "uv": [float(uv[0]), float(uv[1])], "uv_res": list(common.NATIVE),
                            # diagnostics only, export recomputes the anchor from uv
                            "anchor_mode": anchor.get("mode"), "status": anchor.get("status"),
                            "depth_m": anchor.get("z"),
                            "height_over_ground_m": anchor.get("height_over_ground"),
                            "depth_win": {"n_valid": anchor.get("n_valid"),
                                          "spread": anchor.get("spread")},
                            "p_world": anchor.get("p_world"),
                            "rhe_index": e.get("rhe_index"),
                            # label extent per endpoint; REGION_FULL_M = whole segment
                            "region_m": float(sent.get("region_m", REGION_FULL_M)),
                            # rejection arrives as an amendment record
                            "prop": True}
        with s.lock:
            common.append_jsonl(s.labels_path, record)
            s.answered.add(pair["id"])
            if record["dt_ms"]:
                s.times.append(record["dt_ms"])

        reveal = self._reveal(pair, record)
        return self._json({"ok": True, "uuid": record["uuid"], "rhe": reveal,
                           "progress": s.progress()})

    def _reveal(self, pair, record):
        s = self.store
        if s.no_reveal:
            return None
        try:
            a = s.rhe_at(pair["a"], record["a"])
            b = s.rhe_at(pair["b"], record["b"])
        except (FileNotFoundError, OSError):
            return None
        if a is None or b is None:
            return {"covered": False, "mu_a": a and a["mu"], "mu_b": b and b["mu"]}
        t = rhe_mod.compare(a["mu"], b["mu"])
        return {"covered": True, "mu_a": a["mu"], "mu_b": b["mu"],
                "obs_a": a["obs"], "obs_b": b["obs"], "t": t,
                "agree": None if t is None else (t == record["t"])}

    def _undo(self):
        """Retract the last answer with a tombstone (the log is append-only)."""
        s = self.store
        with s.lock:
            live = s.own_records()
            if not live:
                return self._json({"ok": False})
            last = live[-1]
            common.append_jsonl(s.labels_path,
                                {"v": 1, "undo": True, "id": last["id"],
                                 "target": last["uuid"],
                                 "ts": datetime.now(timezone.utc).isoformat()})
            s.reload()
        return self._json({"ok": True, "id": last["id"]})

    def log_message(self, fmt, *a):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="side_experiments/wrizz_labeling/output")
    ap.add_argument("--cfg", default=common.CFG_FILE)
    ap.add_argument("--port", type=int, default=8778)
    ap.add_argument("--split", default=None,
                    help="comma-separated splits to serve this session, e.g. id_train")
    ap.add_argument("--skip-duplicates", action="store_true",
                    help="defer repeat pairs (they must be answered in a different session)")
    ap.add_argument("--sam", default=None, metavar="DEVICE",
                    help="enable click-prompted regions on this device, e.g. cuda:3")
    ap.add_argument("--no-regions", action="store_true",
                    help="point labels only (region_m 0 on every record); used for OOD tiers")
    ap.add_argument("--annotator", default="lm",
                    help="annotator id; the log is filtered to it, so annotators share one queue")
    ap.add_argument("--no-reveal", action="store_true",
                    help="withhold the RHE reveal and path peek (needed for inter-annotator comparison)")
    ap.add_argument("--skip-sources", default=None,
                    help="comma-separated pair_source values to defer, e.g. rhe_overlap")
    args = ap.parse_args()
    Handler.store = Store(os.path.abspath(args.out), args.cfg,
                          splits=set(args.split.split(",")) if args.split else None,
                          skip_dup=args.skip_duplicates,
                          skip_sources=set(args.skip_sources.split(",")) if args.skip_sources else (),
                          annotator=args.annotator)
    Handler.store.sam_device = args.sam
    Handler.store.no_regions = args.no_regions
    Handler.store.no_reveal = args.no_reveal
    if args.no_regions and args.sam:
        raise SystemExit("--sam prompts regions, which --no-regions then discards")
    if args.sam:
        Handler.store.sam()      # load up front
        Handler.store.start_prefetch()
        print(f"  SAM prompting on {args.sam}, prefetching {PREFETCH} pairs ahead", flush=True)
    Handler.store.warm()
    n = len(Handler.store.pairs)
    # count over served pairs only (the log covers every split)
    done = sum(1 for p in Handler.store.pairs if p["id"] in Handler.store.answered)
    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"labeling UI on http://localhost:{args.port}  "
          f"({done}/{n} answered by {args.annotator})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
