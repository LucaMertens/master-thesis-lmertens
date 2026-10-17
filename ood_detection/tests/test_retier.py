"""Tier assignment from tiers.json, and the validation guarding what the UI writes."""

import json

import numpy as np
import pytest

from shared_utils import tiers as retier
from ood_detection.tiering import server

STATE = {
    "tiers": ["id_train", "gravel", "metal"],
    "missions": {
        "EIG-1": [{"start": 0, "end": 20, "tier": "gravel"},
                  {"start": 20, "end": 30, "tier": None},
                  {"start": 30, "end": None, "tier": "metal"}],
        "ALB-3": [{"start": 0, "end": None, "tier": "unused"}],
        "SNOW-1": [{"start": 0, "end": None, "tier": "gravel"}],
    },
}

T0 = 1000.0


@pytest.fixture
def state(monkeypatch):
    monkeypatch.setattr(retier, "state", lambda: STATE)
    monkeypatch.setattr(retier, "t0", lambda m: T0)


def assign(pairs, train=()):
    m, t = zip(*pairs)
    return list(retier.assign(np.array(m), np.array(t, dtype=float), train))


def test_segments_and_gaps(state):
    got = assign([("EIG-1", T0 + t) for t in (0, 19.9, 20, 29.9, 30, 500)])
    assert got == ["gravel", "gravel", "", "", "metal", "metal"]


def test_unused_and_unassigned_beat_the_id_train_fallback(state):
    assert assign([("ALB-3", T0)], train=["ALB-3"]) == [""]
    assert assign([("EIG-1", T0 + 25)], train=["EIG-1"]) == [""]


def test_id_train_fallback_only_for_missions_the_file_omits(state):
    assert assign([("CYN-1", T0), ("SPX-1", T0)], train=["CYN-1"]) == ["id_train", ""]


def test_tier_missions_names_a_mission_once(monkeypatch):
    monkeypatch.setattr(retier, "state", lambda: STATE)
    got = retier.tier_missions(["CYN-1", "EIG-1"])
    assert got == {"id_train": ["CYN-1"], "gravel": ["EIG-1", "SNOW-1"], "metal": ["EIG-1"]}


def test_time_ranges_skips_whole_missions(monkeypatch):
    monkeypatch.setattr(retier, "state", lambda: STATE)
    monkeypatch.setattr(retier, "t0", lambda m: T0)
    assert retier.time_ranges("gravel") == {"EIG-1": [[T0 + 0, T0 + 20]]}
    assert retier.time_ranges("metal") == {"EIG-1": [[T0 + 30, None]]}


def test_derived_views():
    missions = STATE["missions"] | {"EIG-2": [{"start": 0, "end": None, "tier": None}]}
    whole, spans, pending = retier.views(missions)
    assert whole == {"SNOW-1": "gravel"}  # ALB-3 is unused, so it is not a whole-run tier
    assert spans == {"EIG-1": [(0, 20, "gravel"), (30, None, "metal")]}
    assert pending == ["EIG-2"]  # ALB-3 is excluded on purpose, not untriaged


def test_shipped_file_validates(monkeypatch):
    monkeypatch.setattr(server, "manifest", lambda: {
        "missions": {m: {"duration": 1e4} for m in json.loads(
            retier.TIERS_FILE.read_text())["missions"]}})
    saved = json.loads(retier.TIERS_FILE.read_text())
    assert server.validate(saved)["tiers"] == saved["tiers"]


@pytest.mark.parametrize("bad", [
    {"tiers": ["a", "a"], "missions": {}},
    {"tiers": ["a"], "missions": {"EIG-1": [{"start": 5, "end": None, "tier": "a"}]}},
    {"tiers": ["a"], "missions": {"EIG-1": [{"start": 0, "end": 3, "tier": "a"}]}},
    {"tiers": ["a"], "missions": {"EIG-1": [{"start": 0, "end": 3, "tier": "a"},
                                            {"start": 9, "end": None, "tier": "a"}]}},
    {"tiers": ["a"], "missions": {"EIG-1": [{"start": 0, "end": None, "tier": "b"}]}},
    {"tiers": ["a"], "missions": {"NOPE-9": [{"start": 0, "end": None, "tier": "a"}]}},
])
def test_validate_rejects(monkeypatch, bad):
    monkeypatch.setattr(server, "manifest", lambda: {"missions": {"EIG-1": {"duration": 400}}})
    with pytest.raises(ValueError):
        server.validate(bad)
