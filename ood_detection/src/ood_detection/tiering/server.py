"""Drag-and-drop editor for the segment tiers in shared_utils/tiers.json (stdlib server, localhost).

    uv run python -m ood_detection.tiering.server        # then open the printed URL
"""

import argparse
import json
import mimetypes
import os
import re
import shutil
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import yaml

from ood_detection.common.retier import TIERS_FILE, UNUSED, state
from ood_detection.tiering.previews import OUT_ROOT

STATIC = Path(__file__).parent / "static"
# Missions the cached features were extracted for (hardcoded, records what is on disk)
EXTRACTED_AT_LAST_RUN = ["KAEB-1", "ROOT-1", "EIG-1", "EIG-2", "PIL-2", "SNOW-1", "SNOW-2",
                         "SNOW-3", "CON-4", "ETH-2", "HAUS-1", "LEICA-2"]
BACKUPS = OUT_ROOT / "backups"  # off the repo, on /datahome, so a rebuild cannot take them
KEEP_BACKUPS = 50
CONFIG = (Path(__file__).resolve().parents[4]
          / "training/traversability_network/configs/gt_human_mt_ineq075_f2.yaml")


def manifest():
    return json.loads((OUT_ROOT / "manifest.json").read_text())


def config_missions():
    cfg = yaml.safe_load(CONFIG.read_text())["DATASET"]
    return cfg["TRAIN_MISSIONS"], cfg["VALID_MISSIONS"]


def extracted_missions(train, valid):
    """Missions with cached OOD windows; others need re-extraction before they score."""
    out = set(train) | set(valid) | set(EXTRACTED_AT_LAST_RUN)
    return sorted(out)


def payload():
    train, valid = config_missions()
    man = manifest()
    saved = json.loads(TIERS_FILE.read_text())
    cached = extracted_missions(train, valid)
    meta = {m: {"duration": e["duration"], "video": e["video"], "cached": m in cached}
            for m, e in man["missions"].items()}
    return {"tiers": saved["tiers"], "missions": saved["missions"], "meta": meta,
            "speed": man["speed"], "auto_id_train": train, "unused": UNUSED,
            "config": CONFIG.name}


def validate(new):
    """Raises ValueError unless every mission's parts tile [0, duration) in order."""
    tiers = new["tiers"]
    if len(set(tiers)) != len(tiers):
        raise ValueError("duplicate tier name")
    known = set(tiers) | {UNUSED, None}
    durations = {m: e["duration"] for m, e in manifest()["missions"].items()}
    for m, parts in new["missions"].items():
        if m not in durations:
            raise ValueError(f"unknown mission {m}")
        if not parts or parts[0]["start"] != 0 or parts[-1]["end"] is not None:
            raise ValueError(f"{m}: parts must cover the whole mission")
        for a, b in zip(parts, parts[1:]):
            if a["end"] is None or a["end"] != b["start"] or a["end"] <= a["start"]:
                raise ValueError(f"{m}: parts are not contiguous and increasing")
        for p in parts:
            if p["tier"] not in known:
                raise ValueError(f"{m}: unknown tier {p['tier']}")
    return {"tiers": tiers, "missions": dict(sorted(new["missions"].items()))}


def backup():
    """Keeps the last KEEP_BACKUPS versions (the board autosaves)."""
    BACKUPS.mkdir(parents=True, exist_ok=True)
    shutil.copy2(TIERS_FILE, BACKUPS / f"tiers-{time.strftime('%Y%m%d-%H%M%S')}.json")
    for old in sorted(BACKUPS.glob("tiers-*.json"))[:-KEEP_BACKUPS]:
        old.unlink()


def save(new):
    checked = validate(new)
    if json.loads(TIERS_FILE.read_text()) != checked:
        backup()
    tmp = TIERS_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(checked, indent=1) + "\n")
    tmp.replace(TIERS_FILE)
    state.cache_clear()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_):
        pass

    @property
    def route(self):
        # not cached: keep-alive reuses the handler across requests
        return self.path.split("?")[0]

    def send_json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path):
        """Serves `path`, honouring a single Range for seeking."""
        if not path.is_file():
            return self.send_error(404)
        data = path.read_bytes()
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        start, end = 0, len(data) - 1
        rng = re.match(r"bytes=(\d+)-(\d*)", self.headers.get("Range", ""))
        if rng:
            start = int(rng.group(1))
            end = int(rng.group(2)) if rng.group(2) else end
        chunk = data[start:end + 1]
        self.send_response(206 if rng else 200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        if rng:
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
        self.send_header("Content-Length", str(len(chunk)))
        self.send_header("Cache-Control", "no-cache" if ctype.startswith("text") else "max-age=3600")
        self.end_headers()
        self.wfile.write(chunk)

    def do_GET(self):
        if self.route == "/api/state":
            return self.send_json(payload())
        if self.route.startswith("/media/"):
            return self.send_file(OUT_ROOT / os.path.basename(self.route))
        name = "index.html" if self.route == "/" else os.path.basename(self.route)
        self.send_file(STATIC / name)

    def do_PUT(self):
        if self.route != "/api/state":
            return self.send_error(404)
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        try:
            save(json.loads(body))
        except (ValueError, KeyError, TypeError) as e:
            return self.send_json({"error": str(e)}, 400)
        self.send_json({"ok": True})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--host", default="127.0.0.1")
    args = p.parse_args()
    if not (OUT_ROOT / "manifest.json").exists():
        raise SystemExit("no clips yet: uv run python -m ood_detection.tiering.previews")
    print(f"tier editor on http://{args.host}:{args.port}  -> {TIERS_FILE}")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
