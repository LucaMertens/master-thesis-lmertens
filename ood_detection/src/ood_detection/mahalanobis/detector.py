"""Post-hoc feature-distance OOD detector (Mahalanobis++, Ren et al. 2021): the
'simple fix' is standardizing features and using a shrinkage-regularized
covariance instead of the naive empirical inverse, which is unstable once feature
dimension approaches the in-distribution sample count. Feature-space agnostic:
the same class scores WayFASTER's penultimate BEV feature or a DINO feature,
which is what makes the optional RQ4 comparison (WayFASTER-space vs DINO-space
AUROC, identical splits) just two .fit()/.score() calls instead of new code."""

from sklearn.covariance import LedoitWolf


class MahalanobisDetector:
    def fit(self, X):
        self.mean_ = X.mean(0)
        self.std_ = X.std(0) + 1e-6
        self.cov_ = LedoitWolf().fit((X - self.mean_) / self.std_)
        return self

    def score(self, X):
        """Higher = more OOD (further from the in-distribution Gaussian)."""
        return self.cov_.mahalanobis((X - self.mean_) / self.std_)
