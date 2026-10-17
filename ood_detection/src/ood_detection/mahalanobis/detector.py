"""Mahalanobis++ (Mueller & Hein 2025) on L2-normalized features, with k-means or GMM modes as classes."""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.covariance import LedoitWolf
from sklearn.mixture import GaussianMixture

CHUNK = 1 << 20
# Mode counts BIC chooses between (scored on a subsample, refit at the winner)
BIC_GRID = (2, 4, 8, 16, 32, 64, 128)
BIC_FIT = 30_000


def l2_normalize(X):
    X = np.asarray(X, dtype=np.float64)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def top_frac_mean(d, frac=0.1):
    """Mean of the worst `frac` of each row; frac=1 is the plain mean."""
    if frac >= 1:
        return d.mean(-1)
    k = max(1, int(round(frac * d.shape[-1])))
    return np.sort(d, axis=-1)[..., -k:].mean(-1)


METRICS = ("mahalanobis", "euclidean", "cosine")


class MahalanobisPP:
    """Distance to the nearest mode on L2-normalized features.

    `modes`: `kmeans` (k-means centroids, Ledoit-Wolf covariance of residuals) or `gmm`
    (tied-covariance mixture, BIC picks the count).
    `metric`: `mahalanobis` (shared covariance), `euclidean` (plain k-means) or `cosine`."""

    def __init__(self, n_modes=1, top_frac=0.1, max_fit=200_000, seed=0, metric="mahalanobis",
                 modes="kmeans"):
        assert metric in METRICS, metric
        assert modes in ("kmeans", "gmm"), modes
        self.modes = modes
        self.n_modes = n_modes
        self.top_frac = top_frac
        self.max_fit = max_fit
        self.seed = seed
        self.metric = metric

    def fit(self, X):
        """X: (N, D) in-distribution samples, one row per cell."""
        rng = np.random.default_rng(self.seed)
        if len(X) > self.max_fit:
            X = X[rng.choice(len(X), self.max_fit, replace=False)]
        Z = l2_normalize(X)
        cov = self._fit_gmm(Z, rng) if self.modes == "gmm" else self._fit_kmeans(Z)
        self.whiten_ = (np.linalg.cholesky(np.linalg.inv(cov))
                        if self.metric == "mahalanobis" else np.eye(Z.shape[1]))
        self.unit_means_ = self.means_ / (np.linalg.norm(self.means_, axis=1, keepdims=True) + 1e-12)
        return self

    def _fit_kmeans(self, Z):
        km = KMeans(self.n_modes, n_init=10, random_state=self.seed).fit(Z)
        self.means_ = km.cluster_centers_
        self.n_modes_ = self.n_modes
        return LedoitWolf(assume_centered=True).fit(Z - self.means_[km.labels_]).covariance_

    def _fit_gmm(self, Z, rng):
        """Tied-covariance mixture with the component count chosen by BIC on ID data."""
        sub = Z[rng.choice(len(Z), min(BIC_FIT, len(Z)), replace=False)]

        def gm(n, **kw):
            return GaussianMixture(n, covariance_type="tied", random_state=self.seed, **kw)

        self.bic_ = {n: gm(n).fit(sub).bic(sub) for n in BIC_GRID}
        self.n_modes_ = min(self.bic_, key=self.bic_.get)
        fitted = gm(self.n_modes_, n_init=3).fit(Z)
        self.means_ = fitted.means_
        return fitted.covariances_

    def cell_scores(self, X):
        """(N,) distance to the nearest mode. Higher = more OOD."""
        Z = l2_normalize(X)
        out = np.empty(len(Z))
        for i in range(0, len(Z), CHUNK):
            z = Z[i:i + CHUNK]
            if self.metric == "cosine":
                out[i:i + CHUNK] = 1 - (z @ self.unit_means_.T).max(1)
            else:
                d2 = np.stack([(((z - m) @ self.whiten_) ** 2).sum(1) for m in self.means_], 1)
                out[i:i + CHUNK] = np.sqrt(d2.min(1))
        return out

    def frame_scores(self, cells):
        """(B,) score per window from its (B, n_cells, D) cell block."""
        B, n = cells.shape[:2]
        d = self.cell_scores(cells.reshape(B * n, -1)).reshape(B, n)
        return top_frac_mean(d, self.top_frac)
