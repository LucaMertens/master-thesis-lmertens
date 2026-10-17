"""Human W-RIZZ pairwise labels as an additive BEV supervision term, plus the human HDR metric.

One annotation carries one unit of loss mass however many frames saw it."""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R

from .metrics import gather_sum
from .rizz import HEADLINE_TAU, balanced_hdr, hdr_counts, rizz_terms
from .sampling import window_count

# anchor height over ground a BEV-targeted endpoint must sit in (excludes foliage)
HEIGHT_RANGE = (-0.3, 1.2)
# default points per endpoint (anchor + region); the real width is PairTable.n_pts
K_PTS = 9
# residual scale of the occurrence weights (propagate.RESID_ABS)
RESID_ABS = 0.15
# tolerance matching index ts_ns to bag frame times
TS_TOL = 1e-4
# EMA of the headline across validations, for checkpoint selection
EMA_ALPHA = 0.3
# prefix for metrics from human labels alone; "valid/" and "train/" involve RHE labels
NAMESPACE = "human"


def _pairs_from_index(index_root, height_range):
    """Endpoint rows of every mission npz, joined into pairs numbered by label id."""
    rows = {}
    for name in sorted(os.listdir(index_root)):
        if not name.endswith(".npz"):
            continue
        mission = name[:-4]
        d = np.load(os.path.join(index_root, name), allow_pickle=True)
        off = d["offsets"]
        for i, lid in enumerate(d["label_id"]):
            lo, hi = off[i], off[i + 1]
            # the annotated point first, then its region, padded to a fixed width
            n_reg = int(d["n_region"][i])
            pts = np.concatenate([d["p_world"][i][None], d["region_pts"][i][:n_reg]])
            rows.setdefault(str(lid), {})[str(d["side"][i])] = dict(
                mission=mission, ts_ns=int(d["anchor_ts_ns"][i]), pts=pts,
                npts=1 + n_reg, uv=d["anchor_uv"][i].astype(float),
                height=float(d["height"][i]), obs_ts=d["obs_ts_ns"][lo:hi],
                obs_z=d["obs_z"][lo:hi], obs_resid=d["obs_resid"][lo:hi],
                t=int(d["t"][i]), source=str(d["pair_source"][i]), kind=str(d["kind"][i]),
                trainable=bool(d["trainable"][i]), hdr=bool(d["hdr_eligible"][i]),
            )
    out = []
    for lid in sorted(rows):
        e = rows[lid]
        if len(e) != 2:  # an endpoint whose anchor failed takes the comparison with it
            continue
        if not all(height_range[0] <= e[s]["height"] <= height_range[1] for s in "ab"):
            continue
        out.append((lid, e["a"], e["b"]))
    # one padded width for the whole index, so windows stack
    n_pts = max((len(v["pts"]) for _, a, b in out for v in (a, b)), default=K_PTS)
    for _, a, b in out:
        for v in (a, b):
            v["pts"] = np.pad(v["pts"], ((0, n_pts - len(v["pts"])), (0, 0)))
    return out, n_pts


class PairTable:
    """Every usable human pair, in one global numbering."""

    def __init__(self, index_root, height_range=HEIGHT_RANGE):
        self.entries, self.n_pts = _pairs_from_index(index_root, height_range)
        def get(f, conv):
            return np.array([conv(a[f]) for _, a, _ in self.entries])
        self.ids = [lid for lid, _, _ in self.entries]
        self.t = get("t", int).astype(np.int64)
        self.source = get("source", str)
        self.kind = get("kind", str)
        self.trainable = get("trainable", bool)
        self.hdr_eligible = get("hdr", bool)

    def __len__(self):
        return len(self.entries)

    def missions(self, i):
        return (self.entries[i][1]["mission"], self.entries[i][2]["mission"])


def _bag_windows(dataset):
    """(stride, n_windows, present) per bag; ``present`` = window starts the index list visits."""
    visited = set(dataset.indices)
    out = []
    for b, bag in enumerate(dataset.rosbags):
        stride = bag.get("frame_stride", dataset.frame_stride)
        n = window_count(len(bag["image_timestamp"]), dataset.n_frames, stride)
        lo = dataset.bag_sizes[b]
        out.append((stride, n, {w for w in range(n) if lo + w in visited}))
    return out


def _anchor_frames(bag, windows, stride, n_frames):
    """Pose at each window's last input frame, the reference get_states uses."""
    ts = np.asarray(bag["image_timestamp"])
    anchor_ts = ts[np.asarray(windows) + stride * (n_frames - 1)]
    i = np.searchsorted(np.asarray(bag["states_timestamp"]), anchor_ts, "left")
    i = np.clip(i, 0, len(bag["position"]) - 1)
    pos = np.asarray(bag["position"])[i]
    yaw = R.from_quat(np.asarray(bag["quaternion"])[i]).as_euler("ZYX")[:, 0]
    return pos, yaw


def _frame_of(ts, ts_ns):
    """Index of the frame at ``ts_ns`` in a bag's float-second times, or -1."""
    i = int(np.searchsorted(ts, ts_ns / 1e9))
    for j in (i, i - 1):
        if 0 <= j < len(ts) and abs(ts[j] - ts_ns / 1e9) < TS_TOL:
            return j
    return -1


def _to_bev(pts, pos, yaw):
    """World points into one window's BEV frame (x, y), heading-aligned as get_states."""
    d = pts[..., :2] - pos[:2]
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack([c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1]], -1)


class HumanPairs:
    """Per-window human supervision for one dataset split.

    Occurrences are fixed at construction: forkserver workers hold a pickled copy and
    would never see a per-epoch redraw."""

    def __init__(self, table, dataset, grid_bounds, k=5, k_cross=1, anchor_mass=0.4,
                 ineq_fraction=0.5, seed=0, trainable_only=True):
        self.table = table
        self.k = k
        self.k_cross = k_cross
        self.anchor_mass = anchor_mass
        self.n_pts = table.n_pts
        xb, yb = grid_bounds["xbound"], grid_bounds["ybound"]
        rng = np.random.default_rng(seed)

        keep = np.ones(len(table), bool)
        if trainable_only:
            keep &= table.trainable
        mission_set = set(dataset.mission_names)
        keep &= np.array([all(m in mission_set for m in table.missions(i))
                          for i in range(len(table))])

        # mission -> bag indices, and the per-bag window bookkeeping
        by_mission = {}
        for b, bag in enumerate(dataset.rosbags):
            by_mission.setdefault(dataset.mission_names[bag["mission_id"]], []).append(b)
        self._geom = _bag_windows(dataset)
        self._dataset = dataset

        tasks = {}          # (bag, window) -> list of (pair, side, weight, xy, mask)
        self.cross_groups = []
        self.n_used = self.n_cross = 0
        for i in np.flatnonzero(keep):
            occ = [self._endpoint_windows(table.entries[i][1 + s], by_mission, xb, yb)
                   for s in (0, 1)]
            if table.kind[i] == "cross":
                chosen = self._cross_occurrence(occ, rng)
            else:
                chosen = self._intra_occurrences(occ, rng)
            if not chosen:
                continue
            self.n_used += 1
            self.n_cross += table.kind[i] == "cross"
            for w, side_slots in chosen:
                for s, (key, xy, mask) in side_slots.items():
                    tasks.setdefault(key, []).append([i, s, w, xy, mask])
            if table.kind[i] == "cross":
                # one group per occurrence, not per pair: merging a pair's windows makes chains no batch can hold
                self.cross_groups += [tuple(sorted({key for key, _, _ in ss.values()}))
                                      for _, ss in chosen]

        self._apply_class_mass(tasks, ineq_fraction)
        self.window_tasks = {k_: self._pack(v) for k_, v in tasks.items()}
        self.max_per_window = max((len(v[0]) for v in self.window_tasks.values()), default=0)

    # ---- occurrence construction ----

    def _endpoint_windows(self, e, by_mission, xb, yb):
        """Windows that see this endpoint, with the observation each sees it in.

        A window qualifies when an input frame is within half a stride of an observation
        and the annotated point is inside its BEV grid. The slack is needed because
        WINDOW_STRIDE thins windows, so exact hits would drop most observations.

        The annotated frame always counts as an observation of its own point, even if
        propagation dropped it (31% of endpoints). Its depth is not in the index, so it
        is stratified on BEV range instead."""
        n_frames = self._dataset.n_frames
        out = {}
        for b in by_mission.get(e["mission"], ()):
            bag = self._dataset.rosbags[b]
            stride, n_win, present = self._geom[b]
            ts = np.asarray(bag["image_timestamp"])
            f_anchor = _frame_of(ts, e["ts_ns"])
            span = stride * (n_frames - 1)
            slack = stride // 2
            cand = {}
            seen = [(f_anchor, None, 0.0)] if f_anchor >= 0 else []
            seen += [(_frame_of(ts, o), float(z), float(r))
                     for o, z, r in zip(e["obs_ts"], e["obs_z"], e["obs_resid"])]
            for fj, z, resid in seen:
                if fj < 0:
                    continue
                for w in range(fj - span - slack, fj + slack + 1):
                    if not (0 <= w < n_win and w in present):
                        continue
                    # keep the most recent view of the point
                    d = abs(w + span - fj)
                    if w not in cand or d < cand[w][0]:
                        cand[w] = (d, z, resid)
            if not cand:
                continue
            wins = sorted(cand)
            pos, yaw = _anchor_frames(bag, wins, stride, n_frames)
            for idx, w in enumerate(wins):
                xy = _to_bev(e["pts"], pos[idx], yaw[idx])
                inside = ((xy[:, 0] >= xb[0]) & (xy[:, 0] < xb[1])
                          & (xy[:, 1] >= yb[0]) & (xy[:, 1] < yb[1]))
                inside[e["npts"]:] = False
                if not inside[0]:  # the annotated point itself must be in the grid
                    continue
                _, z, resid = cand[w]
                if z is None:
                    z = float(np.hypot(*xy[0]))
                # None where this bag does not hold the annotated frame
                d_anchor = abs(w + span - f_anchor) if f_anchor >= 0 else span + slack + 1
                out[(b, w)] = dict(
                    z=z, resid=resid, xy=xy.astype(np.float32),
                    mask=inside.astype(np.float32),
                    anchor_dist=d_anchor if d_anchor <= span + slack else None)
        return out

    def _slots(self, occ, key_a, key_b):
        a, b = occ[0][key_a], occ[1][key_b]
        return {0: (key_a, a["xy"], a["mask"]), 1: (key_b, b["xy"], b["mask"])}

    def _anchor_key(self, side_occ):
        """The window standing in for the frame the annotator was shown."""
        rated = [(v["anchor_dist"], k) for k, v in side_occ.items() if v["anchor_dist"] is not None]
        return min(rated)[1] if rated else None

    def _cross_occurrence(self, occ, rng):
        """A cross pair scored on ``k_cross`` matched pairs of windows.

        Endpoints are in different recordings, so occurrences come in pairs. Draw i takes
        the i-th window of each side (k window pairs, not |S_a| x |S_b|). Draw 0 is the
        two annotated windows; later draws move both sides together."""
        ka, kb = (self._anchor_key(o) for o in occ)
        if ka is None or kb is None:
            return []
        out = [(1.0, self._slots(occ, ka, kb))]
        rest = [sorted(k for k in o if k != anchor)
                for o, anchor in zip(occ, (ka, kb))]
        n = min(len(rest[0]), len(rest[1]), max(self.k_cross - 1, 0))
        if not n:
            return out
        pick = [rng.permutation(len(r))[:n] for r in rest]
        out[0] = (self.anchor_mass, out[0][1])
        w = (1.0 - self.anchor_mass) / n
        out += [(w, self._slots(occ, rest[0][i], rest[1][j]))
                for i, j in zip(*pick)]
        return out

    def _intra_occurrences(self, occ, rng):
        """k windows that see both endpoints, weights summing to 1.

        The annotated window takes ANCHOR_MASS. The rest is spread over draws from each
        tercile of observation depth, split by depth-residual agreement."""
        shared = sorted(set(occ[0]) & set(occ[1]))
        if not shared:
            return []
        anchor = self._anchor_key({k: occ[0][k] for k in shared})
        if anchor is None:
            anchor = shared[0]
        rest = [k for k in shared if k != anchor]
        if not rest or self.k == 1:
            return [(1.0, self._slots(occ, anchor, anchor))]
        # k = 0: spread the mass over every window that can score the pair
        k = len(shared) if self.k <= 0 else self.k
        out = [(self.anchor_mass, self._slots(occ, anchor, anchor))]
        z = np.array([0.5 * (occ[0][k]["z"] + occ[1][k]["z"]) for k in rest])
        resid = np.array([max(occ[0][k]["resid"], occ[1][k]["resid"]) for k in rest])
        order = np.argsort(z)
        terciles = [list(g) for g in np.array_split(order, 3) if len(g)]
        picked = []
        while len(picked) < k - 1 and any(terciles):
            for g in terciles:
                if not g or len(picked) >= k - 1:
                    continue
                p = 1.0 / (1.0 + resid[g] / RESID_ABS)
                j = g[int(rng.choice(len(g), p=p / p.sum()))]
                g.remove(j)
                picked.append(j)
        w = 1.0 / (1.0 + resid[picked] / RESID_ABS)
        w = (1.0 - out[0][0]) * w / w.sum()
        out += [(float(wi), self._slots(occ, rest[j], rest[j])) for wi, j in zip(w, picked)]
        return out

    # ---- packing ----

    def _apply_class_mass(self, tasks, ineq_fraction):
        """Oversample inequality pairs (55% of answers are ties) via weight instead of copies."""
        self.class_mass = np.ones(2)
        if not ineq_fraction:
            return
        used = {row[0] for v in tasks.values() for row in v}
        t = self.table.t[sorted(used)]
        n_tie, n_ineq = int((t == 0).sum()), int((t != 0).sum())
        if not n_tie or not n_ineq:
            return
        n = n_tie + n_ineq
        self.class_mass = np.array([(1 - ineq_fraction) * n / n_tie,
                                    ineq_fraction * n / n_ineq])
        for rows in tasks.values():
            for row in rows:
                row[2] *= self.class_mass[int(self.table.t[row[0]] != 0)]

    def _pack(self, rows):
        return (np.array([r[0] for r in rows], np.int64),
                np.array([r[1] for r in rows], np.int64),
                np.array([r[2] for r in rows], np.float32),
                np.stack([r[3] for r in rows]),
                np.stack([r[4] for r in rows]))

    # ---- per-sample emission ----

    def emit(self, bag_idx, data_idx, aug, horizontal_flip):
        """Fixed-shape human supervision for one window, in its augmented BEV frame."""
        m = max(self.max_per_window, 1)
        xy = np.zeros((m, self.n_pts, 2), np.float32)
        mask = np.zeros((m, self.n_pts), np.float32)
        pair = np.full(m, -1, np.int64)
        side = np.zeros(m, np.int64)
        weight = np.zeros(m, np.float32)
        t = np.zeros(m, np.float32)
        rows = self.window_tasks.get((bag_idx, data_idx))
        if rows is not None:
            n = len(rows[0])
            p = rows[3].copy()
            if horizontal_flip:
                p[..., 1] = -p[..., 1]
            # same composition as get_states: mirror first, then the yaw/offset aug
            p = p @ np.asarray(aug[:2, :2], np.float32).T + np.asarray(aug[:2, 3], np.float32)
            xy[:n], mask[:n] = p, rows[4]
            pair[:n], side[:n], weight[:n] = rows[0], rows[1], rows[2]
            t[:n] = self.table.t[rows[0]]
        return {"xy": torch.from_numpy(xy), "mask": torch.from_numpy(mask),
                "pair": torch.from_numpy(pair), "side": torch.from_numpy(side),
                "weight": torch.from_numpy(weight), "t": torch.from_numpy(t)}

    def summary(self, name):
        n_occ = sum(len(v[0]) for v in self.window_tasks.values())
        return (f"human pairs [{name}]: {self.n_used}/{len(self.table)} used "
                f"({self.n_cross} cross), {n_occ} window slots over "
                f"{len(self.window_tasks)} windows, max {self.max_per_window} per window, "
                f"tie/ineq mass x{self.class_mass[0]:.2f}/x{self.class_mass[1]:.2f}")


def batch_scores(trav_map, human, grid_bounds):
    """Predicted score per annotated endpoint in a batch, over its region.

    Returns (pair, side, weight, t, score, val, w), keeping endpoints with at least one
    region point inside the BEV grid. ``score`` is the region mean; ``val`` and ``w``
    are the per-point values and validity."""
    xb, yb = grid_bounds["xbound"], grid_bounds["ybound"]
    b, m, p, _ = human["xy"].shape
    xy = human["xy"].reshape(b, m * p, 2).to(trav_map.dtype)
    idxu = 2 * (xy[..., 0] - xb[0]) / (xb[1] - xb[0]) - 1
    idxv = 2 * (xy[..., 1] - yb[0]) / (yb[1] - yb[0]) - 1
    grid = torch.stack((-idxv, -idxu), -1).unsqueeze(1)
    inside = (grid.abs() <= 1).all(-1).squeeze(1).reshape(b, m, p)
    val = F.grid_sample(trav_map[:, :1], grid, align_corners=True)[:, 0, 0].reshape(b, m, p)

    w = human["mask"].to(val.dtype) * inside
    n = w.sum(-1)
    ok = (human["pair"] >= 0) & (n > 0)
    score = (val * w).sum(-1) / n.clamp_min(1)
    return (human["pair"][ok], human["side"][ok], human["weight"][ok].to(val.dtype),
            human["t"][ok].to(val.dtype), score[ok], val[ok], w[ok])


def _join_sides(pair, side, weight, t, score, val, wpt, n_pairs):
    """Pairs whose two endpoints are both scored in this batch.

    Region points are averaged over windows only, never over neighbours. fp32
    throughout: bf16 accumulation of hinge terms loses too much precision."""
    weight, t, score = weight.float(), t.float(), score.float()
    val, wpt = val.float(), wpt.float()
    k = val.shape[-1]
    flat = pair * 2 + side
    acc = torch.zeros(2 * n_pairs, dtype=score.dtype, device=score.device)
    cnt = torch.zeros_like(acc)
    acc = acc.index_add(0, flat, score)
    cnt = cnt.index_add(0, flat, torch.ones_like(score))
    meta = torch.zeros(2 * n_pairs, 2, dtype=score.dtype, device=score.device)
    meta = meta.index_add(0, flat, torch.stack([weight, t], -1))
    pacc = torch.zeros(2 * n_pairs, k, dtype=score.dtype, device=score.device)
    pcnt = torch.zeros_like(pacc)
    pacc = pacc.index_add(0, flat, val * wpt)
    pcnt = pcnt.index_add(0, flat, wpt)
    acc, cnt, meta = acc.view(-1, 2), cnt.view(-1, 2), meta.view(-1, 2, 2)
    pacc, pcnt = pacc.view(-1, 2, k), pcnt.view(-1, 2, k)
    have = (cnt > 0).all(-1)
    idx = have.nonzero().squeeze(1)
    p = (acc[idx] / cnt[idx].clamp_min(1))
    pts = pacc[idx] / pcnt[idx].clamp_min(1e-6)
    w = (meta[idx, :, 0] / cnt[idx].clamp_min(1)).mean(-1)
    t_pair = (meta[idx, 0, 1] / cnt[idx, 0].clamp_min(1))
    return p[:, 0], p[:, 1], pts, pcnt[idx] > 0, t_pair, w, idx


def _point_pair_terms(pts, ok, t, margin, n_samples=0):
    """Mean hinge over every valid (point in A, point in B) combination, and the combination count.

    Whether the combinations count as one example or |A| x |B| is HUMAN_POINT_MASS,
    applied by the caller. ``n_samples`` draws that many cell pairs per batch instead
    of using all of them."""
    if n_samples > 0:
        w = ok.to(pts.dtype)
        w = torch.where(w.sum(-1, keepdim=True) > 0, w, torch.ones_like(w))
        idx = [torch.multinomial(w[:, s], n_samples, replacement=True) for s in (0, 1)]
        a = pts[:, 0].gather(1, idx[0])
        b = pts[:, 1].gather(1, idx[1])
        n = torch.full_like(t, float(n_samples))
        return rizz_terms(a, b, t[:, None], margin).mean(1), n
    a, b = pts[:, 0].unsqueeze(2), pts[:, 1].unsqueeze(1)
    m = (ok[:, 0].unsqueeze(2) & ok[:, 1].unsqueeze(1)).to(pts.dtype)
    n = m.sum((1, 2))
    terms = rizz_terms(a, b, t[:, None, None], margin)
    return (terms * m).sum((1, 2)) / n.clamp_min(1e-6), n


def human_loss(trav_map, human, grid_bounds, margin, n_pairs, point_pairs=False,
               point_mass="annotation", point_samples=0):
    """Weighted squared-hinge (W-RIZZ Eq. 2) over the human pairs in a batch, as a weighted mean.

    Cross pairs contribute only when both annotated windows are in the batch."""
    scored = batch_scores(trav_map, human, grid_bounds)
    p_a, p_b, pts, ok, t, w, idx = _join_sides(*scored, n_pairs)
    # endpoints whose partner is missing; for cross pairs the sampler failed to pin them
    n_half = int(scored[0].unique().numel() - idx.numel())
    if p_a.numel() == 0:
        return trav_map.sum() * 0.0, 0, n_half
    if point_pairs:
        per, n_combo = _point_pair_terms(pts, ok, t, margin, point_samples)
        # "combination": weight by surviving cell pairs; "annotation": one unit per answer
        if point_mass == "combination":
            w = w * n_combo
    else:
        per = rizz_terms(p_a, p_b, t, margin)
    return (w * per).sum() / w.sum().clamp_min(1e-6), int(p_a.numel()), n_half


class HumanHdrMetrics(nn.Module):
    """HDR over the human pairs, balanced between ties and inequalities.

    Endpoints accumulate over the whole validation epoch, so cross-recording pairs count.
    Restricted to ``pair_source == 'random'``; hunt and rhe_overlap pairs are not representative."""

    def __init__(self, table, taus=(0.1, HEADLINE_TAU, 0.5)):
        super().__init__()
        self.eligible = torch.from_numpy(table.source == "random")
        self.register_buffer("t", torch.from_numpy(table.t), persistent=False)
        self.register_buffer("taus", torch.tensor(taus), persistent=False)
        self.register_buffer("acc", torch.zeros(len(table), 2), persistent=False)
        self.register_buffer("cnt", torch.zeros(len(table), 2), persistent=False)
        # per region point, for the point-pair view of the same answers
        self.register_buffer("pacc", torch.zeros(len(table), 2, table.n_pts), persistent=False)
        self.register_buffer("pcnt", torch.zeros(len(table), 2, table.n_pts), persistent=False)
        self.ema = None

    def reset(self):
        self.acc.zero_()
        self.cnt.zero_()
        self.pacc.zero_()
        self.pcnt.zero_()

    @torch.no_grad()
    def update(self, trav_map, human, grid_bounds):
        pair, side, _, _, score, val, w = batch_scores(trav_map, human, grid_bounds)
        if pair.numel() == 0:
            return
        flat = pair * 2 + side
        self.acc.view(-1).index_add_(0, flat, score.float())
        self.cnt.view(-1).index_add_(0, flat, torch.ones_like(score, dtype=torch.float))
        k = self.pacc.shape[-1]
        self.pacc.view(-1, k).index_add_(0, flat, (val * w).float())
        self.pcnt.view(-1, k).index_add_(0, flat, w.float())

    def log(self, pl_module, split="valid"):
        """Also logs an EMA of the headline (raw sd 0.010 vs 0.021 gain over a run), for CKPT_MONITOR."""
        prefix = f"{NAMESPACE}/{split}"
        acc = gather_sum(pl_module, self.acc).clone()
        cnt = gather_sum(pl_module, self.cnt).clone()
        pacc = gather_sum(pl_module, self.pacc).clone()
        pcnt = gather_sum(pl_module, self.pcnt).clone()
        have = (cnt > 0).all(-1) & self.eligible.to(cnt.device)
        idx = have.nonzero().squeeze(1)
        self.reset()
        if idx.numel() == 0:
            return
        p = acc[idx] / cnt[idx]
        counts = hdr_counts(p[:, 0], p[:, 1], self.t[idx], self.taus)
        for i, tau in enumerate(self.taus.tolist()):
            if abs(tau - HEADLINE_TAU) < 1e-9:
                # headline is the balanced rate (55% of answers are ties)
                headline = balanced_hdr(counts[i])
                pl_module.log(f"{prefix}_hdr", headline, sync_dist=False)
                self.ema = headline if self.ema is None else (
                    EMA_ALPHA * headline + (1 - EMA_ALPHA) * self.ema)
                pl_module.log(f"{prefix}_hdr_ema", self.ema, sync_dist=False)
            for j, group in enumerate(("", "_eq", "_ineq")):
                wrong, total = counts[i, j]
                if total > 0:
                    pl_module.log(f"{prefix}_hdr{group}_{tau:g}", wrong / total,
                                  sync_dist=False)
        pl_module.log(f"{prefix}_pairs_scored", float(idx.numel()), sync_dist=False)
        self._log_point_pairs(pl_module, prefix, idx, pacc, pcnt)

    def _log_point_pairs(self, pl_module, prefix, idx, pacc, pcnt):
        """The same answers scored per cell: a pair's disagreement is the share of its
        |A| x |B| cell combinations ordered against the human. Annotations count once each."""
        ok = pcnt[idx] > 0
        pts = pacc[idx] / pcnt[idx].clamp_min(1e-6)
        m = (ok[:, 0].unsqueeze(2) & ok[:, 1].unsqueeze(1))
        n = m.sum((1, 2))
        keep = n > 0
        if not keep.any():
            return
        d = pts[:, 1].unsqueeze(1) - pts[:, 0].unsqueeze(2)
        t = self.t[idx].to(d.dtype)
        groups = [torch.ones_like(t, dtype=torch.bool) & keep, (t == 0) & keep, (t != 0) & keep]
        for tau in self.taus.tolist():
            wrong = ((torch.sign(d) * (d.abs() > tau)) != t[:, None, None]) & m
            frac = wrong.sum((1, 2)) / n.clamp_min(1)
            nw = wrong.sum((1, 2)).to(frac.dtype)
            for tag, block in (("_pts", [frac, torch.ones_like(frac)]),
                               ("_ptsall", [nw, n.to(frac.dtype)])):
                counts = torch.stack([torch.stack([block[0][g].sum(), block[1][g].sum()])
                                      for g in groups])
                if abs(tau - HEADLINE_TAU) < 1e-9:
                    pl_module.log(f"{prefix}_hdr{tag}", balanced_hdr(counts), sync_dist=False)
                for j, group in enumerate(("", "_eq", "_ineq")):
                    if counts[j, 1] > 0:
                        pl_module.log(f"{prefix}_hdr{tag}{group}_{tau:g}",
                                      counts[j, 0] / counts[j, 1], sync_dist=False)


def attach_human_pairs(configs, train_dataset, valid_dataset):
    """Attach human supervision to both splits; returns the train batch sampler.

    The valid split gets one occurrence per annotation (its annotated window)."""
    if configs.TRAINING.HUMAN_WEIGHT <= 0:
        return None
    assert len(configs.TRAINING.DEVICES) == 1, (
        "the human batch sampler emits one epoch, not one shard per rank; run "
        "HUMAN_WEIGHT on a single device")
    assert isinstance(train_dataset, type(valid_dataset)), (
        "human pairs index windows by (bag, window start); a concatenated train set "
        "renumbers them")
    table = PairTable(configs.DATASET.HUMAN_INDEX_ROOT)
    grid = configs.MODEL.GRID_BOUNDS
    train_dataset.human = HumanPairs(
        table, train_dataset, grid, k=configs.TRAINING.HUMAN_K,
        k_cross=configs.TRAINING.HUMAN_K_CROSS,
        anchor_mass=configs.TRAINING.HUMAN_ANCHOR_MASS,
        ineq_fraction=configs.TRAINING.HUMAN_INEQ_FRACTION, seed=configs.SEED)
    valid_dataset.human = HumanPairs(
        table, valid_dataset, grid, k=1, ineq_fraction=0.0, seed=configs.SEED,
        trainable_only=False)
    print(train_dataset.human.summary("train"))
    print(valid_dataset.human.summary("valid"))
    sampler = HumanBatchSampler(train_dataset, train_dataset.human.cross_groups,
                                configs.TRAINING.BATCHSIZE, seed=configs.SEED)
    print(f"human batch sampler: {sampler.n_pinned} cross occurrences pinned into a "
          f"batch, {sampler.n_dropped} dropped (chain longer than BATCHSIZE), "
          f"{len(sampler)} batches")
    return sampler


class HumanBatchSampler(torch.utils.data.Sampler):
    """Shuffled batches that keep each cross pair's two annotated windows in one batch."""

    def __init__(self, dataset, groups, batch_size, drop_last=True, seed=0):
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0
        pos = {}
        for i, w in enumerate(dataset.indices):
            pos.setdefault(w, []).append(i)
        # union-find, so a window shared by two cross pairs pulls both into one group
        parent = list(range(len(dataset.indices)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        # merges chain; drop groups that would exceed the batch size
        size = dict.fromkeys(range(len(parent)), 1)
        self.n_pinned = self.n_dropped = 0
        for g in groups:
            keys = [dataset.bag_sizes[b] + w for b, w in g]
            assert all(k in pos for k in keys), (
                "a cross pair was scheduled onto a window the index list does not "
                "visit; its two endpoints could never meet in a batch")
            roots = {find(pos[k][0]) for k in keys}
            if len(roots) < 2:
                continue
            if sum(size[r] for r in roots) > batch_size:
                self.n_dropped += 1
                continue
            first, *rest = sorted(roots)
            for r in rest:
                parent[r] = first
                size[first] += size[r]
            self.n_pinned += 1
        merged = {}
        for i in range(len(parent)):
            merged.setdefault(find(i), []).append(i)
        self.groups = list(merged.values())
        assert max(len(g) for g in self.groups) <= batch_size
        n = len(dataset.indices)
        self.n_batches = n // batch_size if drop_last else -(-n // batch_size)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        """Pinned groups first, then singles, keeping batch size fixed."""
        rng = np.random.default_rng(self.seed + self.epoch)
        order = rng.permutation(len(self.groups))
        pinned = [self.groups[i] for i in order if len(self.groups[i]) > 1]
        singles = [self.groups[i][0] for i in order if len(self.groups[i]) == 1]
        batches = [[] for _ in range(max(self.n_batches, 1))]
        b = 0
        for g in pinned:
            for _ in range(len(batches)):
                if len(batches[b]) + len(g) <= self.batch_size:
                    break
                b = (b + 1) % len(batches)
            assert len(batches[b]) + len(g) <= self.batch_size, (
                "more pinned cross-pair windows than the epoch has batch room for")
            batches[b] += g
            b = (b + 1) % len(batches)
        it = iter(singles)
        for batch in batches:
            while len(batch) < self.batch_size:
                try:
                    batch.append(next(it))
                except StopIteration:
                    break
            if batch and (len(batch) == self.batch_size or not self.drop_last):
                yield batch

    def __len__(self):
        return self.n_batches
