// Tier board: one card per mission part, columns are tiers. State is the same shape
// tiers.json holds, so the board is a direct editor of the file retier.py reads.
const $ = (q) => document.querySelector(q);
const UNASSIGNED = " unassigned";
let S, saveTimer;
const undo = [];

const fmt = (t) => `${String(Math.floor(t / 60)).padStart(2, "0")}:${String(Math.round(t % 60)).padStart(2, "0")}`;
const dur = (m) => S.meta[m].duration;
const parts = (m) => S.missions[m] ?? [{ start: 0, end: null, tier: auto(m) ? "id_train" : null }];
const auto = (m) => !(m in S.missions) && S.auto_id_train.includes(m);
const endOf = (m, p) => (p.end === null ? dur(m) : p.end);
const colOf = (p) => (p.tier === null ? UNASSIGNED : p.tier);

function hue(s) { let h = 0; for (const c of s) h = (h * 31 + c.charCodeAt(0)) % 360; return h; }

function color(name) {
  if (name === UNASSIGNED) return "#555b68";
  if (name === S.unused) return "#5a2b2b";
  return `hsl(${hue(name)} 45% 42%)`;
}

const label = (name) => (name === UNASSIGNED ? "unassigned" : name);
// a decimal below 10 min, since a short tier's balance turns on a quarter minute
const mins = (secs) => (secs < 600 ? (secs / 60).toFixed(1) : Math.round(secs / 60));

// -- mutations ---------------------------------------------------------------
function snapshot() { undo.push(JSON.stringify({ tiers: S.tiers, missions: S.missions })); }

function materialize(m) { if (!(m in S.missions)) S.missions[m] = structuredClone(parts(m)); }

function changed() { render(); scheduleSave(); }

function assignPart(m, i, tier) {
  snapshot();
  materialize(m);
  S.missions[m][i].tier = tier;
  changed();
}

function splitAt(m, t) {
  materialize(m);
  const ps = S.missions[m];
  const i = ps.findIndex((p) => t > p.start && t < endOf(m, p));
  if (i < 0) return false;
  snapshot();
  ps.splice(i + 1, 0, { start: t, end: ps[i].end, tier: ps[i].tier });
  ps[i].end = t;
  changed();
  return true;
}

function mergeIntoPrevious(m, i) {
  if (i < 1) return false;
  snapshot();
  materialize(m);
  const ps = S.missions[m];
  ps[i - 1].end = ps[i].end;
  ps.splice(i, 1);
  changed();
  return true;
}

function addTier() {
  const name = prompt("new tier name");
  if (!name || S.tiers.includes(name)) return;
  snapshot();
  S.tiers.push(name);
  changed();
}

function renameTier(old) {
  const name = prompt("rename tier", old);
  if (!name || name === old || S.tiers.includes(name)) return;
  snapshot();
  S.tiers[S.tiers.indexOf(old)] = name;
  for (const ps of Object.values(S.missions)) for (const p of ps) if (p.tier === old) p.tier = name;
  changed();
}

function deleteTier(name) {
  const n = Object.values(S.missions).flat().filter((p) => p.tier === name).length;
  if (n && !confirm(`${name} holds ${n} part(s); move them to unassigned?`)) return;
  snapshot();
  S.tiers.splice(S.tiers.indexOf(name), 1);
  for (const ps of Object.values(S.missions)) for (const p of ps) if (p.tier === name) p.tier = null;
  changed();
}

function moveTier(name, d) {
  const i = S.tiers.indexOf(name);
  const j = i + d;
  if (j < 0 || j >= S.tiers.length) return;
  snapshot();
  S.tiers.splice(j, 0, ...S.tiers.splice(i, 1));
  changed();
}

// -- persistence -------------------------------------------------------------
function status(text, bad) {
  const e = $("#status");
  e.textContent = text;
  e.style.color = bad ? "#e0736b" : "";
}

function scheduleSave() {
  status("saving");
  clearTimeout(saveTimer);
  saveTimer = setTimeout(async () => {
    const r = await fetch("/api/state", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ tiers: S.tiers, missions: S.missions }),
    });
    const d = await r.json();
    status(r.ok ? `saved ${new Date().toLocaleTimeString()}` : `not saved: ${d.error}`, !r.ok);
  }, 400);
}

// -- cards -------------------------------------------------------------------
const cache = new Map();

const observer = new IntersectionObserver(
  (es) => es.forEach((e) => {
    e.target.dataset.vis = e.isIntersecting ? "1" : "";
    syncPlay(e.target);
  }),
  { threshold: 0.05 },
);

function syncPlay(v) {
  if ($("#play-all").checked && v.dataset.vis) v.play().catch(() => {});
  else v.pause();
}

function buildCard(m) {
  const el = document.createElement("div");
  el.className = "card";
  el.draggable = true;
  el.innerHTML = `<video muted playsinline preload="metadata" src="/media/${S.meta[m].video}"></video>
    <div class="badge"></div>
    <div class="tools"><button class="split" title="split / inspect">cut</button>
      <button class="unuse" title="mark unused">x</button></div>
    <div class="meta"><div class="name">${m}</div><div class="span"></div></div>`;
  const v = el.querySelector("video");
  v.addEventListener("timeupdate", () => {
    const s = +v.dataset.s;
    const e = +v.dataset.e;
    if (v.currentTime >= e || v.currentTime < s - 0.25) v.currentTime = s;
  });
  el.addEventListener("dragstart", (ev) =>
    ev.dataTransfer.setData("text/plain", `${el.dataset.mission}\n${el.dataset.index}`));
  el.querySelector(".split").onclick = () => openEditor(el.dataset.mission, +el.dataset.index);
  el.querySelector(".unuse").onclick = () => assignPart(el.dataset.mission, +el.dataset.index, S.unused);
  observer.observe(v);
  return el;
}

function cardEl(m, i, p, isAuto) {
  const key = `${m}@${p.start}`;
  let el = cache.get(key);
  if (!el) cache.set(key, (el = buildCard(m)));
  el.dataset.mission = m;
  el.dataset.index = i;
  el.classList.toggle("auto", isAuto);
  el.classList.toggle("stale", !S.meta[m].cached);
  el.querySelector(".badge").textContent = isAuto ? "auto" : (S.meta[m].cached ? "" : "no cache");
  const whole = p.start === 0 && p.end === null;
  el.querySelector(".span").textContent = whole ? `${fmt(dur(m))} full` : `${fmt(p.start)}-${fmt(endOf(m, p))}`;
  const v = el.querySelector("video");
  v.dataset.s = p.start / S.speed;
  v.dataset.e = endOf(m, p) / S.speed;
  if (v.currentTime < +v.dataset.s || v.currentTime > +v.dataset.e) v.currentTime = +v.dataset.s;
  return el;
}

// -- board -------------------------------------------------------------------
const tierOfColumn = (name) => (name === UNASSIGNED ? null : name);

function makeColumn(name) {
  const col = document.createElement("div");
  col.className = "col";
  const special = name === UNASSIGNED || name === S.unused;
  const controls = special ? "" : `<button class="l" title="move left">&lt;</button>
      <button class="r" title="move right">&gt;</button>
      <button class="x" title="delete tier">x</button>`;
  col.innerHTML = `<div class="head"><span class="swatch" style="background:${color(name)}"></span>
      <span class="name" title="double-click to rename">${label(name)}</span>
      <span class="count dim"></span>${controls}</div><div class="body"></div>`;
  if (!special) {
    col.querySelector(".name").ondblclick = () => renameTier(name);
    col.querySelector(".l").onclick = () => moveTier(name, -1);
    col.querySelector(".r").onclick = () => moveTier(name, 1);
    col.querySelector(".x").onclick = () => deleteTier(name);
  }
  col.addEventListener("dragover", (e) => { e.preventDefault(); col.classList.add("over"); });
  col.addEventListener("dragleave", () => col.classList.remove("over"));
  col.addEventListener("drop", (e) => {
    e.preventDefault();
    col.classList.remove("over");
    const [m, i] = e.dataTransfer.getData("text/plain").split("\n");
    assignPart(m, +i, tierOfColumn(name));
  });
  return col;
}

function visible(m) {
  const q = $("#filter").value.trim().toLowerCase();
  return (!q || m.toLowerCase().includes(q)) && (!$("#cached-only").checked || S.meta[m].cached);
}

function render() {
  const board = $("#board");
  board.replaceChildren();
  const cols = {};
  for (const name of [UNASSIGNED, ...S.tiers, S.unused]) board.append((cols[name] = makeColumn(name)));
  const n = {}, secs = {};
  for (const m of Object.keys(S.meta).sort()) {
    const isAuto = auto(m);
    parts(m).forEach((p, i) => {
      const name = cols[colOf(p)] ? colOf(p) : UNASSIGNED;
      // totals ignore the filter: they are what the tier holds, not what is on screen
      n[name] = (n[name] ?? 0) + 1;
      secs[name] = (secs[name] ?? 0) + endOf(m, p) - p.start;
      if (visible(m)) cols[name].querySelector(".body").append(cardEl(m, i, p, isAuto));
    });
  }
  for (const [name, col] of Object.entries(cols))
    col.querySelector(".count").textContent = `${n[name] ?? 0} / ${mins(secs[name] ?? 0)} min`;
  document.querySelectorAll(".card video").forEach(syncPlay);
}

// -- editor ------------------------------------------------------------------
const ed = { m: null, sel: 0 };
let scrubbing = false;

function openEditor(m, i) {
  ed.m = m;
  ed.sel = i;
  const v = $("#ed-video");
  if (!v.src.endsWith(S.meta[m].video)) v.src = `/media/${S.meta[m].video}`;
  $("#ed-scrub").max = dur(m);
  $("#editor").hidden = false;
  const start = () => { seek(parts(m)[i].start); renderEditor(); v.play().catch(() => {}); };
  if (v.readyState) start();
  else v.addEventListener("loadedmetadata", start, { once: true });
}

function closeEditor() {
  $("#editor").hidden = true;
  $("#ed-video").pause();
  ed.m = null;
}

function renderEditor() {
  const m = ed.m;
  const ps = parts(m);
  ed.sel = Math.min(ed.sel, ps.length - 1);
  $("#ed-title").textContent = `${m} / ${fmt(dur(m))} / ${S.speed}x`;
  const strip = $("#ed-strip");
  strip.replaceChildren();
  ps.forEach((p, i) => {
    const b = document.createElement("div");
    b.className = "band" + (i === ed.sel ? " sel" : "");
    b.style.left = `${(100 * p.start) / dur(m)}%`;
    b.style.width = `${(100 * (endOf(m, p) - p.start)) / dur(m)}%`;
    b.style.background = color(colOf(p));
    b.textContent = p.tier ?? "";
    b.title = `${fmt(p.start)}-${fmt(endOf(m, p))} ${p.tier ?? "unassigned"}`;
    b.onclick = () => { ed.sel = i; seek(p.start); renderEditor(); };
    strip.append(b);
  });
  const head = document.createElement("div");
  head.className = "head";
  strip.append(head);
  const sel = ps[ed.sel];
  $("#ed-part").textContent = `part ${ed.sel + 1}/${ps.length}  ${fmt(sel.start)}-${fmt(endOf(m, sel))}`;
  const t = $("#ed-tier");
  t.replaceChildren();
  for (const name of ["", ...S.tiers, S.unused]) {
    const o = document.createElement("option");
    o.value = name;
    o.textContent = name === "" ? "(unassigned)" : name;
    t.append(o);
  }
  t.value = sel.tier ?? "";
  updateHead();
}

function updateHead() {
  const v = $("#ed-video");
  if (!ed.m) return;
  const t = v.currentTime * S.speed;
  const head = $("#ed-strip .head");
  if (head) head.style.left = `${(100 * t) / dur(ed.m)}%`;
  $("#ed-time").textContent = fmt(t);
  if (!scrubbing) $("#ed-scrub").value = t;
}

function seek(t) { $("#ed-video").currentTime = t / S.speed; }

function togglePlay() {
  const v = $("#ed-video");
  if (v.paused) v.play().catch(() => {});
  else v.pause();
}

function initEditor() {
  const v = $("#ed-video");
  for (const e of ["play", "pause"]) v.addEventListener(e, () => {
    $("#ed-play").textContent = v.paused ? "play" : "pause";
  });
  $("#ed-play").onclick = togglePlay;
  v.onclick = togglePlay;
  v.addEventListener("timeupdate", () => {
    updateHead();
    if (!$("#ed-part-only").checked || !ed.m) return;
    const p = parts(ed.m)[ed.sel];
    const t = v.currentTime * S.speed;
    if (t >= endOf(ed.m, p) || t < p.start - 1) seek(p.start);
  });
  $("#ed-scrub").addEventListener("input", (e) => { scrubbing = true; seek(+e.target.value); });
  $("#ed-scrub").addEventListener("change", () => { scrubbing = false; });
  $("#ed-close").onclick = closeEditor;
  $("#ed-split").onclick = () => {
    const t = Math.round(v.currentTime * S.speed);
    if (splitAt(ed.m, t)) {
      ed.sel = parts(ed.m).findIndex((p) => p.start === t);
      renderEditor();
    }
  };
  $("#ed-merge").onclick = () => { if (mergeIntoPrevious(ed.m, ed.sel)) { ed.sel--; renderEditor(); } };
  $("#ed-tier").onchange = (e) => { assignPart(ed.m, ed.sel, e.target.value || null); renderEditor(); };
  $("#editor").addEventListener("click", (e) => {
    if (e.target.id === "editor") closeEditor();
    // keep focus off the buttons, or space re-fires the last one instead of playing
    if (e.target.tagName === "BUTTON") e.target.blur();
  });
  addEventListener("keydown", (e) => {
    if (e.key === "z" && (e.ctrlKey || e.metaKey)) {
      if (!undo.length) return;
      Object.assign(S, JSON.parse(undo.pop()));
      changed();
      if (ed.m) renderEditor();
      return;
    }
    if (!ed.m || e.target.tagName === "INPUT") return;
    if (e.key === "Escape") closeEditor();
    if (e.key === " ") { e.preventDefault(); togglePlay(); }
    if (e.key === "ArrowRight") seek(v.currentTime * S.speed + (e.shiftKey ? 10 : 1));
    if (e.key === "ArrowLeft") seek(v.currentTime * S.speed - (e.shiftKey ? 10 : 1));
    if (e.key === "s") $("#ed-split").click();
  });
}

// -- boot --------------------------------------------------------------------
fetch("/api/state").then((r) => r.json()).then((d) => {
  S = d;
  $("#speed").textContent = `${S.speed}x  ${S.config}`;
  $("#add-tier").onclick = addTier;
  $("#filter").oninput = render;
  $("#cached-only").onchange = render;
  $("#play-all").onchange = () => document.querySelectorAll(".card video").forEach(syncPlay);
  initEditor();
  render();
});
