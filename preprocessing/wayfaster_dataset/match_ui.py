#!/usr/bin/env python
"""Build a local web UI that georeferences the BEV mosaic against satellite imagery."""

import argparse
import base64
import io
import pathlib

import matplotlib
import numpy as np
from shared_utils.paths import FIGURES_ROOT

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CENTER = (40.102824225665536, -88.22719583725755)

HTML = """<!doctype html>
<meta charset="utf-8">
<title>WayFASTER georeference</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; font: 13px/1.45 system-ui, sans-serif; background: #14161a; color: #e6e8ea; }}
  header {{ padding: 8px 12px; background: #1c1f24; border-bottom: 1px solid #2a2f36;
            display: flex; gap: 16px; align-items: center; flex-wrap: wrap; }}
  header b {{ font-weight: 600; }}
  #wrap {{ display: grid; grid-template-columns: 1fr 1fr; height: calc(100vh - 152px); }}
  #left, #right {{ position: relative; }}
  #left {{ border-right: 1px solid #2a2f36; background: #fff; }}
  canvas {{ display: block; width: 100%; height: 100%; cursor: crosshair; }}
  #map {{ width: 100%; height: 100%; }}
  footer {{ height: 100px; padding: 8px 12px; background: #1c1f24; border-top: 1px solid #2a2f36;
            display: flex; gap: 12px; }}
  #list {{ flex: 1; overflow: auto; font-family: ui-monospace, monospace; font-size: 12px; }}
  #list div {{ padding: 1px 0; }}
  .side {{ width: 260px; }}
  button {{ font: inherit; padding: 4px 10px; margin-right: 6px; border-radius: 5px;
            border: 1px solid #39404a; background: #262b32; color: #e6e8ea; cursor: pointer; }}
  button:hover {{ background: #2f353d; }}
  .tag {{ padding: 2px 7px; border-radius: 4px; background: #262b32; }}
  .warn {{ color: #ffb454; }} .ok {{ color: #8fd06f; }} .dim {{ color: #8b939d; }}
  label {{ display: flex; gap: 6px; align-items: center; }}
</style>

<header>
  <b>1.</b> click the {left_label} <b>2.</b> click the same spot on the map
  <label>rotate <input id="rot" type="range" min="-180" max="180" value="0" step="1"><span id="rotv" class="tag">0&deg;</span></label>
  <span id="status" class="tag dim">0 pairs</span>
  <span id="fit" class="tag dim">need 2 pairs</span>
</header>

<div id="wrap">
  <div id="left"><canvas id="cv"></canvas></div>
  <div id="right"><div id="map"></div></div>
</div>

<footer>
  <div id="list"></div>
  <div class="side">
    <button onclick="undo()">Undo</button>
    <button onclick="clearAll()">Clear</button>
    <button onclick="copyOut()">Copy JSON</button>
    <div class="dim" style="margin-top:6px">{left_label}: drag pan, wheel zoom</div>
  </div>
</footer>

<script>
const EXTENT = {extent};              // [x0, x1, y0, y1] in metres
const CENTER = {center};
const IMG = new Image();
IMG.src = "data:image/png;base64,{b64}";

const cv = document.getElementById('cv'), ctx = cv.getContext('2d');
let view = {{ zoom: 1, px: 0, py: 0, rot: 0 }};
let pairs = [], pending = null;

const map = L.map('map').setView(CENTER, 18);
L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}',
  {{ maxZoom: 21, maxNativeZoom: 19, attribution: 'Imagery: Esri, Maxar, Earthstar Geographics' }}).addTo(map);
let mapMarkers = [];

function resize() {{
  cv.width = cv.clientWidth * devicePixelRatio;
  cv.height = cv.clientHeight * devicePixelRatio;
  draw();
}}
addEventListener('resize', resize);
IMG.onload = () => {{ view.zoom = 1; resize(); }};

function matrix() {{
  const m = new DOMMatrix();
  m.translateSelf(cv.width / 2 + view.px, cv.height / 2 + view.py);
  m.rotateSelf(view.rot);
  const base = Math.min(cv.width / IMG.width, cv.height / IMG.height);
  m.scaleSelf(base * view.zoom, base * view.zoom);
  m.translateSelf(-IMG.width / 2, -IMG.height / 2);
  return m;
}}

function px2m(ix, iy) {{
  return [EXTENT[0] + ix / IMG.width * (EXTENT[1] - EXTENT[0]),
          EXTENT[3] - iy / IMG.height * (EXTENT[3] - EXTENT[2])];
}}
function m2px(x, y) {{
  return [(x - EXTENT[0]) / (EXTENT[1] - EXTENT[0]) * IMG.width,
          (EXTENT[3] - y) / (EXTENT[3] - EXTENT[2]) * IMG.height];
}}

function draw() {{
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, cv.width, cv.height);
  if (!IMG.complete) return;
  const m = matrix();
  ctx.setTransform(m);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(IMG, 0, 0);
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  const marks = pairs.map((p, i) => [p.bev, String.fromCharCode(65 + i)]);
  if (pending) marks.push([pending, '?']);
  for (const [bev, lab] of marks) {{
    const [ix, iy] = m2px(bev[0], bev[1]);
    const pt = m.transformPoint(new DOMPoint(ix, iy));
    ctx.beginPath(); ctx.arc(pt.x, pt.y, 7 * devicePixelRatio, 0, 7);
    ctx.fillStyle = lab === '?' ? '#ffb454' : '#e5484d'; ctx.fill();
    ctx.lineWidth = 2 * devicePixelRatio; ctx.strokeStyle = '#fff'; ctx.stroke();
    ctx.fillStyle = '#fff'; ctx.font = `${{11 * devicePixelRatio}}px sans-serif`;
    ctx.fillText(lab, pt.x + 9 * devicePixelRatio, pt.y - 7 * devicePixelRatio);
  }}
}}

let drag = null;
cv.addEventListener('mousedown', e => drag = {{ x: e.offsetX, y: e.offsetY, moved: 0, px: view.px, py: view.py }});
addEventListener('mouseup', () => drag = null);
cv.addEventListener('mousemove', e => {{
  if (!drag) return;
  const dx = e.offsetX - drag.x, dy = e.offsetY - drag.y;
  drag.moved = Math.max(drag.moved, Math.abs(dx) + Math.abs(dy));
  view.px = drag.px + dx * devicePixelRatio;
  view.py = drag.py + dy * devicePixelRatio;
  draw();
}});
cv.addEventListener('wheel', e => {{
  e.preventDefault();
  const f = e.deltaY < 0 ? 1.15 : 1 / 1.15;
  view.zoom *= f;
  view.px *= f; view.py *= f;
  draw();
}}, {{ passive: false }});

cv.addEventListener('click', e => {{
  if (drag && drag.moved > 4) return;
  const p = matrix().inverse().transformPoint(
    new DOMPoint(e.offsetX * devicePixelRatio, e.offsetY * devicePixelRatio));
  if (p.x < 0 || p.y < 0 || p.x > IMG.width || p.y > IMG.height) return;
  pending = px2m(p.x, p.y);
  render();
}});

document.getElementById('rot').addEventListener('input', e => {{
  view.rot = +e.target.value;
  document.getElementById('rotv').textContent = view.rot + '\\u00b0';
  draw();
}});

map.on('click', e => {{
  if (!pending) {{ flash('click the {left_label} first'); return; }}
  pairs.push({{ bev: pending, ll: [e.latlng.lat, e.latlng.lng] }});
  pending = null;
  render();
}});

function flash(msg) {{
  const s = document.getElementById('status');
  s.textContent = msg; s.className = 'tag warn';
  setTimeout(render, 1200);
}}
function undo() {{ if (pending) pending = null; else pairs.pop(); render(); }}
function clearAll() {{ pairs = []; pending = null; render(); }}
function copyOut() {{
  navigator.clipboard.writeText(JSON.stringify(pairs.map(p => ({{
    {key}: [round(p.bev[0], 1), round(p.bev[1], 1)], lat: p.ll[0], lon: p.ll[1] }})), null, 1));
  flash('copied');
}}
const round = (v, n) => Math.round(v * 10 ** n) / 10 ** n;

// local ENU metres about CENTER
function enu(lat, lon) {{
  const R = 111319.49;
  return [(lon - CENTER[1]) * Math.cos(CENTER[0] * Math.PI / 180) * R, (lat - CENTER[0]) * R];
}}

// 2D similarity: enu ~= s*R*bev + t
function fitSim() {{
  const n = pairs.length;
  if (n < 2) return null;
  const A = pairs.map(p => p.bev), B = pairs.map(p => enu(p.ll[0], p.ll[1]));
  const ca = [0, 1].map(k => A.reduce((s, v) => s + v[k], 0) / n);
  const cb = [0, 1].map(k => B.reduce((s, v) => s + v[k], 0) / n);
  let ns = 0, nc = 0, va = 0;
  for (let i = 0; i < n; i++) {{
    const ax = A[i][0] - ca[0], ay = A[i][1] - ca[1];
    const bx = B[i][0] - cb[0], by = B[i][1] - cb[1];
    ns += ax * by - ay * bx; nc += ax * bx + ay * by; va += ax * ax + ay * ay;
  }}
  const th = Math.atan2(ns, nc), s = Math.hypot(ns, nc) / va;
  const c = Math.cos(th), si = Math.sin(th);
  const res = pairs.map((p, i) => {{
    const ax = A[i][0] - ca[0], ay = A[i][1] - ca[1];
    const ex = cb[0] + s * (c * ax - si * ay), ey = cb[1] + s * (si * ax + c * ay);
    return Math.hypot(ex - B[i][0], ey - B[i][1]);
  }});
  const rms = Math.sqrt(res.reduce((a, b) => a + b * b, 0) / n);
  return {{ deg: th * 180 / Math.PI, scale: s, rms, res }};
}}

function render() {{
  mapMarkers.forEach(m => map.removeLayer(m));
  mapMarkers = pairs.map((p, i) => L.circleMarker(p.ll, {{
    radius: 6, color: '#fff', weight: 2, fillColor: '#e5484d', fillOpacity: 1
  }}).addTo(map).bindTooltip(String.fromCharCode(65 + i), {{ permanent: true, direction: 'right' }}));

  const f = fitSim();
  const el = document.getElementById('list');
  el.innerHTML = pairs.map((p, i) => {{
    const r = f ? f.res[i] : null;
    const flagged = r !== null && pairs.length > 2 && r > 3 * f.rms && r > 2;
    return `<div class="${{flagged ? 'warn' : ''}}">${{String.fromCharCode(65 + i)}}: `
      + `(${{round(p.bev[0], 1)}}, ${{round(p.bev[1], 1)}}) -> ${{p.ll[0].toFixed(6)}}, ${{p.ll[1].toFixed(6)}}`
      + (r !== null ? `  <span class="dim">res ${{r.toFixed(2)}} m</span>` : '') + `</div>`;
  }}).join('') || '<div class="dim">no pairs yet</div>';

  const s = document.getElementById('status');
  s.className = 'tag dim';
  s.textContent = pending ? 'now click the map' : pairs.length + ' pairs';

  const fe = document.getElementById('fit');
  if (!f) {{ fe.textContent = 'need 2 pairs'; fe.className = 'tag dim'; }}
  else {{
    fe.textContent = `rot ${{f.deg.toFixed(1)}}\\u00b0  scale ${{f.scale.toFixed(4)}}  rms ${{f.rms.toFixed(2)}} m`;
    fe.className = 'tag ' + (pairs.length >= 3 && f.rms < 3 && Math.abs(f.scale - 1) < 0.05 ? 'ok' : 'warn');
  }}
  draw();
}}
render();
</script>
"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=pathlib.Path,
                   default=FIGURES_ROOT / "cache/wayfaster_overlap.npz")
    p.add_argument("--out", type=pathlib.Path, default=pathlib.Path("/tmp/wayfaster_match.html"))
    p.add_argument("--image", type=pathlib.Path,
                   help="match an arbitrary image instead of the mosaic; coordinates "
                        "are then pixels (x right, y up from the bottom left)")
    args = p.parse_args()

    if args.image:
        b64 = base64.b64encode(args.image.read_bytes()).decode()
        shape = plt.imread(args.image).shape[:2]
        extent = [0.0, float(shape[1]), 0.0, float(shape[0])]
        key, label = "px", "image"
    else:
        d = np.load(args.cache)
        rgb, cnt, extent = d["rgb"], d["cnt"], d["extent"]
        filled = cnt > 0
        img = np.full(rgb.shape, 255, np.uint8)
        img[filled] = np.clip(rgb[filled] / cnt[filled, None], 0, 255).astype(np.uint8)
        buf = io.BytesIO()
        plt.imsave(buf, np.flipud(img), format="png")
        b64 = base64.b64encode(buf.getvalue()).decode()
        extent = [round(float(v), 3) for v in extent]
        shape, key, label = img.shape[:2], "bev", "mosaic"

    args.out.write_text(HTML.format(
        extent=extent, center=list(CENTER), b64=b64, key=key, left_label=label))
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB), "
          f"{label} {shape[1]}x{shape[0]}, coords in {key}")


if __name__ == "__main__":
    main()
