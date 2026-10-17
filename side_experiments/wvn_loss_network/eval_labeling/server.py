"""Backend server for the eval labeling UI."""

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


class Handler(BaseHTTPRequestHandler):
    out = None  # set in main

    def _frames(self):
        return json.load(open(os.path.join(self.out, "frames.json")))

    def _entry(self, name):
        return next((e for e in self._frames() if e["name"] == name), None)

    def _send(self, code, body=b"", ctype="application/octet-stream"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path, ctype):
        if not os.path.exists(path):
            return self._send(404, b"not found", "text/plain")
        with open(path, "rb") as f:
            self._send(200, f.read(), ctype)

    def do_GET(self):
        p = self.path.split("?")[0]
        if p == "/":
            return self._send_file(os.path.join(HERE, "static", "index.html"), "text/html")
        if p == "/api/frames":
            frames = self._frames()
            for e in frames:
                e["labeled"] = os.path.exists(os.path.join(self.out, "labels", e["name"] + ".png"))
            return self._send(200, json.dumps(frames).encode(), "application/json")
        for prefix, sub, ctype in (("/frames/", "frames", "image/jpeg"), ("/segments/", "segments", "image/png")):
            if p.startswith(prefix):
                name = os.path.splitext(os.path.basename(p))[0]
                if self._entry(name) is None:
                    return self._send(404, b"unknown frame", "text/plain")
                ext = ".jpg" if sub == "frames" else ".png"
                return self._send_file(os.path.join(self.out, sub, name + ext), ctype)
        if p.startswith("/api/labels/"):
            name = os.path.basename(p)
            if self._entry(name) is None:
                return self._send(404, b"unknown frame", "text/plain")
            path = os.path.join(self.out, "labels", name + ".png")
            if not os.path.exists(path):
                return self._send(204)
            arr = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            return self._send(200, arr.tobytes())
        self._send(404, b"not found", "text/plain")

    def do_POST(self):
        p = self.path.split("?")[0]
        if not p.startswith("/api/labels/"):
            return self._send(404, b"not found", "text/plain")
        name = os.path.basename(p)
        e = self._entry(name)
        if e is None:
            return self._send(404, b"unknown frame", "text/plain")
        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n)
        if len(body) != e["w"] * e["h"]:
            return self._send(400, b"bad length", "text/plain")
        arr = np.frombuffer(body, np.uint8).reshape(e["h"], e["w"])
        path = os.path.join(self.out, "labels", name + ".png")
        if arr.any():
            os.makedirs(os.path.dirname(path), exist_ok=True)
            cv2.imwrite(path, arr)
        elif os.path.exists(path):
            os.remove(path)  # all-unknown = unlabeled
        self._send(200, b"ok", "text/plain")

    def log_message(self, fmt, *a):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="side_experiments/wvn_loss_network/output/eval_labeling")
    ap.add_argument("--port", type=int, default=8777)
    args = ap.parse_args()
    Handler.out = os.path.abspath(args.out)
    srv = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"labeling UI on http://localhost:{args.port} (data: {Handler.out})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
