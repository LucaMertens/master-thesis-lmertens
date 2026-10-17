import numpy as np
import pytest

from ood_detection.mahalanobis.detector import MahalanobisPP, l2_normalize, top_frac_mean

DIM = 8


def anisotropic(rng, n, shift=0.0, dim=DIM):
    """A tight anisotropic cone around e0, shifted off-axis by `shift` (zero-mean clouds spread over the sphere)."""
    x = rng.normal(0, 1, (n, dim)) @ np.diag(np.linspace(1.0, 0.15, dim))
    x[:, 0] += 8.0
    return x + shift * np.eye(dim)[1]


def test_l2_normalize_gives_unit_rows():
    X = np.random.default_rng(0).normal(0, 3, (50, DIM))
    assert np.allclose(np.linalg.norm(l2_normalize(X), axis=1), 1.0)


def test_top_frac_mean_reads_the_worst_tail():
    d = np.tile(np.arange(100.0), (3, 1))
    assert np.allclose(top_frac_mean(d, 0.1), np.arange(90, 100).mean())
    assert np.allclose(top_frac_mean(d, 1.0), d.mean(1))


def test_shifted_cells_score_higher():
    rng = np.random.default_rng(0)
    det = MahalanobisPP(n_modes=1).fit(anisotropic(rng, 4000))
    id_s = det.cell_scores(anisotropic(rng, 500))
    ood_s = det.cell_scores(anisotropic(rng, 500, shift=2.0))
    assert ood_s.mean() > id_s.mean()


def test_modes_capture_multimodal_id():
    """A point between two ID modes is OOD for k=2 and ID for k=1."""
    rng = np.random.default_rng(1)
    X = np.vstack([anisotropic(rng, 3000, shift=-4.0), anisotropic(rng, 3000, shift=4.0)])
    middle = anisotropic(rng, 300, shift=0.0)
    scores = {k: MahalanobisPP(n_modes=k).fit(X).cell_scores(middle).mean() for k in (1, 2)}
    assert scores[2] > scores[1]


def test_frame_score_ignores_a_clean_majority():
    """One anomalous decile lifts the frame score; the same cells averaged do not."""
    rng = np.random.default_rng(2)
    det = MahalanobisPP(n_modes=1, top_frac=0.1).fit(anisotropic(rng, 4000))
    clean = anisotropic(rng, 200)[None]
    mixed = clean.copy()
    mixed[0, :20] = anisotropic(rng, 20, shift=3.0)
    lift = det.frame_scores(mixed) - det.frame_scores(clean)
    pooled = det.cell_scores(mixed[0]).mean() - det.cell_scores(clean[0]).mean()
    assert lift > pooled


@pytest.mark.parametrize("n_modes", [1, 3])
def test_scores_are_finite_and_chunk_invariant(n_modes, monkeypatch):
    rng = np.random.default_rng(3)
    det = MahalanobisPP(n_modes=n_modes).fit(anisotropic(rng, 2000))
    X = anisotropic(rng, 300)
    full = det.cell_scores(X)
    monkeypatch.setattr("ood_detection.mahalanobis.detector.CHUNK", 64)
    assert np.isfinite(full).all()
    assert np.allclose(full, det.cell_scores(X))
