"""Principal component analysis down to two dimensions, for plotting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PCAProjection:
    """A fitted 2-D PCA. ``transform`` maps vectors of the fitted space to
    plot coordinates; the fitted data itself lands inside ``[-1, 1]``."""

    mean: np.ndarray  # (d,)
    components: np.ndarray  # (2, d), orthonormal rows
    scale: float
    explained_variance_ratio: tuple[float, float]

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        return ((x - self.mean) @ self.components.T) / self.scale


def fit_pca_2d(x: np.ndarray) -> PCAProjection:
    """Fits PCA via an eigendecomposition of the ``d x d`` covariance
    matrix (cheaper than an SVD of the data when ``n >> d``, which is the
    usual case for a knowledge base).

    Component signs are fixed (largest-magnitude loading positive), so the
    same data always produces the same picture instead of randomly
    mirrored ones between runs.
    """
    x = np.asarray(x, dtype=np.float64)
    n, d = x.shape
    mean = x.mean(axis=0) if n else np.zeros(d)
    components = np.zeros((2, d))
    ratio = (0.0, 0.0)
    if n >= 2:
        centered = x - mean
        cov = centered.T @ centered / (n - 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvals = np.clip(eigvals[order], 0.0, None)
        eigvecs = eigvecs[:, order]
        k = min(2, d)
        components[:k] = eigvecs[:, :k].T
        for i in range(k):
            pivot = int(np.argmax(np.abs(components[i])))
            if components[i, pivot] < 0:
                components[i] *= -1
        total = float(eigvals.sum())
        if total > 0:
            top = [float(v) / total for v in eigvals[:2]] + [0.0, 0.0]
            ratio = (top[0], top[1])
    coords = (x - mean) @ components.T if n else np.zeros((0, 2))
    scale = float(np.abs(coords).max()) if coords.size else 0.0
    return PCAProjection(
        mean=mean,
        components=components,
        scale=scale if scale > 1e-12 else 1.0,
        explained_variance_ratio=ratio,
    )
