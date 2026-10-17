"""HDR summary over human pairs, against a direct numpy computation."""

import numpy as np
from traversability_network.adaptations import human_eval


def _reference(p_a, p_b, t, tau):
    d = p_b - p_a
    said = np.sign(d) * (np.abs(d) > tau)
    wrong = said != t
    eq, ineq = t == 0, t != 0
    return wrong[eq].mean(), wrong[ineq].mean()


def test_summary_matches_direct_rates():
    rng = np.random.default_rng(0)
    p_a, p_b = rng.random(200), rng.random(200)
    t = rng.choice([-1.0, 0.0, 1.0], 200)
    s = human_eval.summary(p_a, p_b, t)
    for tau in human_eval.TAUS:
        eq, ineq = _reference(p_a, p_b, t, tau)
        assert np.isclose(s[f"hdr_eq_{tau:g}"], eq, atol=1e-6)
        assert np.isclose(s[f"hdr_ineq_{tau:g}"], ineq, atol=1e-6)
        assert np.isclose(s[f"hdr_balanced_{tau:g}"], 0.5 * (eq + ineq), atol=1e-6)


def test_best_threshold_is_the_scan_minimum():
    rng = np.random.default_rng(1)
    t = rng.choice([-1.0, 0.0, 1.0], 300)
    p_a = rng.random(300)
    p_b = np.clip(p_a + 0.3 * t + 0.05 * rng.standard_normal(300), 0, 1)
    s = human_eval.summary(p_a, p_b, t)
    curve = [np.mean(_reference(p_a, p_b, t, tau)) for tau in human_eval.TAU_SCAN]
    assert np.isclose(s["hdr_balanced_best"], min(curve), atol=1e-6)
    assert s["hdr_balanced_best"] <= s["hdr_balanced_0.25"] + 1e-9


def test_constant_prediction_is_chance():
    t = np.array([-1.0, 0.0, 1.0, 0.0])
    p = np.full(4, 0.7)
    s = human_eval.summary(p, p, t)
    assert np.isclose(s["hdr_balanced_0.25"], 0.5)
