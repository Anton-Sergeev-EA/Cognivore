"""k-means++ clustering with the number of clusters chosen automatically by
silhouette score -- the "how many topics does this knowledge base have?"
question, answered from the data instead of a hard-coded constant."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class KMeansResult:
    labels: np.ndarray  # (n,) int
    centroids: np.ndarray  # (k, d)
    inertia: float


def _sq_dists(x: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    # ||x - c||^2 = ||x||^2 - 2 x.c + ||c||^2, clipped against round-off.
    d = (x * x).sum(1)[:, None] - 2.0 * x @ centroids.T + (centroids * centroids).sum(1)[None, :]
    return np.maximum(d, 0.0)


def _kmeans_pp_init(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = x.shape[0]
    centroids = [x[int(rng.integers(n))]]
    closest = _sq_dists(x, np.array(centroids))[:, 0]
    for _ in range(1, k):
        total = float(closest.sum())
        # total == 0: every point already coincides with a centroid.
        idx = int(rng.integers(n)) if total <= 0 else int(rng.choice(n, p=closest / total))
        centroids.append(x[idx])
        closest = np.minimum(closest, _sq_dists(x, x[idx : idx + 1])[:, 0])
    return np.array(centroids)


def kmeans(
    x: np.ndarray, k: int, seed: int = 0, n_init: int = 4, max_iter: int = 100
) -> KMeansResult:
    """Lloyd's algorithm with k-means++ seeding, best of ``n_init`` runs.
    Deterministic for a given ``seed``."""
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    if n == 0:
        raise ValueError("kmeans needs at least one point")
    k = max(1, min(k, n))
    best: KMeansResult | None = None
    for run in range(n_init):
        rng = np.random.default_rng(seed + run)
        centroids = _kmeans_pp_init(x, k, rng)
        labels = np.zeros(n, dtype=np.int64)
        for it in range(max_iter):
            new_labels = _sq_dists(x, centroids).argmin(1)
            if it > 0 and np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for c in range(k):
                members = x[labels == c]
                if len(members):
                    centroids[c] = members.mean(0)
        inertia = float(_sq_dists(x, centroids)[np.arange(n), labels].sum())
        if best is None or inertia < best.inertia:
            best = KMeansResult(labels=labels.copy(), centroids=centroids.copy(), inertia=inertia)
    assert best is not None
    return best


def silhouette_score(x: np.ndarray, labels: np.ndarray) -> float:
    """Mean silhouette coefficient (Rousseeuw, 1987), in ``[-1, 1]``.
    O(n^2) memory -- callers subsample large inputs first."""
    x = np.asarray(x, dtype=np.float64)
    labels = np.asarray(labels)
    unique = np.unique(labels)
    n = len(labels)
    if len(unique) < 2 or len(unique) >= n:
        return 0.0
    dist = np.sqrt(_sq_dists(x, x))
    scores = np.zeros(n)
    for i in range(n):
        same = labels == labels[i]
        n_same = int(same.sum())
        if n_same <= 1:
            continue  # singleton clusters score 0 by definition
        a = dist[i, same].sum() / (n_same - 1)
        b = min(dist[i, labels == other].mean() for other in unique if other != labels[i])
        denom = max(a, b)
        scores[i] = (b - a) / denom if denom > 0 else 0.0
    return float(scores.mean())


def auto_kmeans(
    x: np.ndarray, k_min: int = 2, k_max: int = 8, seed: int = 0, sample_size: int = 800
) -> tuple[KMeansResult, float]:
    """Runs k-means for every ``k`` in ``[k_min, k_max]`` and keeps the one
    with the best silhouette score. Returns ``(result, silhouette)``.

    Tiny inputs (fewer than 4 points) are returned as a single cluster:
    there is nothing meaningful to separate.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    if n < 4:
        return kmeans(x, 1, seed=seed, n_init=1), 0.0
    rng = np.random.default_rng(seed)
    sample = np.sort(rng.choice(n, size=min(n, sample_size), replace=False))
    best: tuple[KMeansResult, float] | None = None
    for k in range(k_min, min(k_max, n - 1) + 1):
        result = kmeans(x, k, seed=seed)
        score = silhouette_score(x[sample], result.labels[sample])
        if best is None or score > best[1] + 1e-9:
            best = (result, score)
    assert best is not None
    return best
