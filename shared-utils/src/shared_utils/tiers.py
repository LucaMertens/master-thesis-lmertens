"""Evaluation tiers per mission (or stretch of one), read from tiers.json.

tiers.json holds relative mm:ss parts per mission (zero is the first labeled frame);
tier null or "unused" drops the windows. Functions here return absolute seconds.
Unmentioned missions fall back to id_train if the checkpoint trained on them.
Edit with `python -m ood_detection.tiering.server`.
"""

import functools
import json
from pathlib import Path

import numpy as np

from shared_utils.paths import LABELS_ROOT

TIERS_FILE = Path(__file__).with_name("tiers.json")
UNUSED = "unused"  # excluded on purpose, as opposed to a part nobody has looked at yet
ID_TRAIN = "id_train"


@functools.cache
def state(path=TIERS_FILE):
    return json.loads(Path(path).read_text())


def tiers():
    """Tier names in reporting order."""
    return list(state()["tiers"])


def parts(mission=None):
    """Every mission's parts, or one mission's (empty if the file omits it)."""
    missions = state()["missions"]
    return missions if mission is None else missions.get(mission, [])


@functools.cache
def t0(mission):
    """Zero of the mm:ss clock: the mission's first labeled frame."""
    return float(np.load(LABELS_ROOT / mission / "rhe" / "rhe_timestamp.npy", mmap_mode="r")[0])


def _assigned(ps):
    return [p for p in ps if p["tier"] not in (None, UNUSED)]


def views(missions):
    """(whole-run tiers, per-mission spans, untriaged missions); used by figure and video scripts."""
    whole = {m: p[0]["tier"] for m, p in missions.items()
             if len(p) == 1 and p[0]["tier"] not in (None, UNUSED)}
    spans = {m: [(p["start"], p["end"], p["tier"]) for p in _assigned(ps)]
             for m, ps in missions.items() if len(ps) > 1 and _assigned(ps)}
    pending = [m for m, ps in missions.items() if all(p["tier"] is None for p in ps)]
    return whole, spans, pending


def tier_missions(id_train_missions=()):
    """tier -> the missions feeding it, each named once however many spans it gives."""
    out = {t: [] for t in tiers()}
    for m in sorted(id_train_missions):
        if not parts(m):  # unmentioned, so it falls back to id_train whole
            out[ID_TRAIN].append(m)
    for m, ps in sorted(parts().items()):
        for p in ps:
            if p["tier"] in out and m not in out[p["tier"]]:
                out[p["tier"]].append(m)
    return {t: sorted(ms) for t, ms in out.items() if ms}


def time_ranges(tier):
    """mission -> [[start, end], ...] in absolute seconds for `tier`.

    Whole-recording and unmentioned missions are omitted (datasets read that as "use all").
    """
    out = {}
    for m, ps in sorted(parts().items()):
        spans = [p for p in ps if p["tier"] == tier]
        if not spans or (len(ps) == 1 and len(spans) == 1):
            continue
        out[m] = [[t0(m) + p["start"],
                   None if p["end"] is None else t0(m) + p["end"]] for p in spans]
    return out


def split_missions():
    """Missions cut into more than one part, whatever the parts were assigned."""
    return sorted(m for m, ps in parts().items() if len(ps) > 1)


def assign(mission, timestamp, id_train_missions):
    """(N,) tier name per window, "" where the window belongs to no tier."""
    mission = np.asarray(mission)
    timestamp = np.asarray(timestamp, dtype=np.float64)
    out = np.full(len(mission), "", dtype=object)
    for m in id_train_missions:
        out[mission == m] = ID_TRAIN
    for m, ps in parts().items():
        in_mission = mission == m
        if not in_mission.any():
            continue
        rel = timestamp - t0(m)
        for p in ps:
            end = np.inf if p["end"] is None else p["end"]
            tier = p["tier"] if p["tier"] not in (None, UNUSED) else ""
            out[in_mission & (rel >= p["start"]) & (rel < end)] = tier
    return out
