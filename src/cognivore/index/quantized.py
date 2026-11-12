"""Batch-built, compressed cosine indexes with bounded query scratch space."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .python_index import SearchResult

_BATCH = 4096


def _matrix(values: ArrayLike) -> NDArray[np.float32]:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or not matrix.shape[1]:
        raise ValueError("expected a matrix with at least one vector dimension")
    return matrix


def _unit_rows(matrix: NDArray[np.float32]) -> NDArray[np.float32]:
    if not np.isfinite(matrix).all():
        raise ValueError("vectors must be finite")
    norms = np.sqrt(np.einsum("ij,ij->i", matrix, matrix, dtype=np.float64))
    normalized = np.zeros_like(matrix)
    np.divide(matrix, norms[:, None], out=normalized, where=norms[:, None] != 0)
    return normalized


def _query_vector(query: ArrayLike, dim: int) -> NDArray[np.float32]:
    vector = np.asarray(query, dtype=np.float32)
    if vector.shape != (dim,):
        raise ValueError(f"expected a query of length {dim}")
    return _unit_rows(vector[None, :])[0]


def _ids(values: ArrayLike, count: int) -> NDArray[np.int64]:
    labels = np.asarray(values)
    if labels.shape != (count,) or (count and labels.dtype.kind not in "iu"):
        raise ValueError("expected one integer ID per vector")
    if labels.dtype.kind == "u" and np.any(labels > np.iinfo(np.int64).max):
        raise ValueError("IDs must fit in signed 64-bit integers")
    return labels.astype(np.int64, copy=True)


def _inverse_norms(squared: NDArray[np.float32]) -> NDArray[np.float32]:
    norms = np.sqrt(squared)
    return np.divide(1.0, norms, out=np.zeros_like(norms), where=norms != 0)


def _top_results(ids: NDArray[np.int64], scores: NDArray[np.float32], k: int) -> list[SearchResult]:
    k = min(k, len(ids))
    if k <= 0:
        return []
    np.clip(scores, -1.0, 1.0, out=scores)
    selected = np.argpartition(-scores, k - 1)[:k]
    ordered = selected[np.argsort(-scores[selected], kind="stable")]
    return [SearchResult(int(ids[i]), float(scores[i])) for i in ordered]


class ScalarQuantizedIndex:
    """SQ8 with one signed byte per coordinate and a per-vector cosine factor."""

    def __init__(self, ids: ArrayLike, vectors: ArrayLike) -> None:
        matrix = _matrix(vectors)
        self.dim = matrix.shape[1]
        self._ids = _ids(ids, len(matrix))
        self._codes = np.empty(matrix.shape, dtype=np.int8)
        self._inverse_norms = np.empty(len(matrix), dtype=np.float32)
        for start in range(0, len(matrix), _BATCH):
            rows = slice(start, start + _BATCH)
            unit = _unit_rows(matrix[rows])
            scale = np.max(np.abs(unit), axis=1, keepdims=True) / 127.0
            codes = np.divide(unit, scale, out=unit, where=scale != 0)
            np.rint(codes, out=codes)
            self._codes[rows] = codes.astype(np.int8)
            self._inverse_norms[rows] = _inverse_norms(np.einsum("ij,ij->i", codes, codes))

    def __len__(self) -> int:
        return len(self._ids)

    @property
    def storage_bytes(self) -> int:
        """Bytes in the owned arrays, including IDs and cosine factors."""
        return self._ids.nbytes + self._codes.nbytes + self._inverse_norms.nbytes

    def search(self, query: ArrayLike, k: int) -> list[SearchResult]:
        vector = _query_vector(query, self.dim)
        if k <= 0:
            return []
        scores = np.empty(len(self), dtype=np.float32)
        for start in range(0, len(self), _BATCH):
            rows = slice(start, start + _BATCH)
            scores[rows] = (self._codes[rows] @ vector) * self._inverse_norms[rows]
        return _top_results(self._ids, scores, k)


def _nearest(points: NDArray[np.float32], centers: NDArray[np.float32]) -> NDArray[np.intp]:
    assignments = np.empty(len(points), dtype=np.intp)
    penalty = np.einsum("ij,ij->i", centers, centers)
    for start in range(0, len(points), _BATCH):
        rows = slice(start, start + _BATCH)
        assignments[rows] = np.argmin(penalty - 2.0 * (points[rows] @ centers.T), axis=1)
    return assignments


def _train(
    points: NDArray[np.float32], clusters: int, iterations: int, rng: np.random.Generator
) -> NDArray[np.float32]:
    centers = points[rng.choice(len(points), clusters, replace=False)].copy()
    for _ in range(iterations):
        assignments = _nearest(points, centers)
        counts = np.bincount(assignments, minlength=clusters)
        sums = np.zeros_like(centers)
        np.add.at(sums, assignments, points)
        present = counts > 0
        updated = centers.copy()
        updated[present] = sums[present] / counts[present, None]
        if np.array_equal(updated, centers):
            break
        centers = updated
    return centers


class ProductQuantizedIndex:
    """PQ with deterministic subspace codebooks and asymmetric cosine search."""

    def __init__(
        self,
        ids: ArrayLike,
        vectors: ArrayLike,
        *,
        subspaces: int = 8,
        clusters: int = 256,
        iterations: int = 12,
        max_train: int = 12_000,
        seed: int = 42,
        training_vectors: ArrayLike | None = None,
    ) -> None:
        matrix = _matrix(vectors)
        self.dim = matrix.shape[1]
        self._ids = _ids(ids, len(matrix))
        if not 1 <= subspaces <= self.dim or self.dim % subspaces:
            raise ValueError("subspaces must be a positive divisor of the dimension")
        if not 2 <= clusters <= 256 or iterations < 1 or max_train < clusters:
            raise ValueError(
                "expected 2-256 clusters, positive iterations and max_train >= clusters"
            )
        training = matrix if training_vectors is None else _matrix(training_vectors)
        if training.shape[1] != self.dim or len(training) < clusters:
            raise ValueError(
                "training requires at least one vector per cluster and matching dimensions"
            )

        rng = np.random.default_rng(seed)
        if len(training) > max_train:
            training = training[rng.choice(len(training), max_train, replace=False)]
        training = _unit_rows(training)
        width = self.dim // subspaces
        self._books = np.empty((subspaces, clusters, width), dtype=np.float32)
        for part, book in enumerate(self._books):
            dims = slice(part * width, (part + 1) * width)
            book[:] = _train(training[:, dims], clusters, iterations, rng)

        self._codes = np.empty((len(matrix), subspaces), dtype=np.uint8)
        self._inverse_norms = np.empty(len(matrix), dtype=np.float32)
        center_norms = np.einsum("scd,scd->sc", self._books, self._books)
        for start in range(0, len(matrix), _BATCH):
            rows = slice(start, start + _BATCH)
            unit = _unit_rows(matrix[rows])
            squared = np.zeros(len(unit), dtype=np.float32)
            for part, book in enumerate(self._books):
                dims = slice(part * width, (part + 1) * width)
                codes = _nearest(unit[:, dims], book)
                self._codes[rows, part] = codes
                squared += center_norms[part, codes]
            squared[~unit.any(axis=1)] = 0.0
            self._inverse_norms[rows] = _inverse_norms(squared)

    def __len__(self) -> int:
        return len(self._ids)

    @property
    def storage_bytes(self) -> int:
        """Bytes in the owned codes, codebooks, IDs and cosine factors."""
        return (
            self._ids.nbytes + self._codes.nbytes + self._inverse_norms.nbytes + self._books.nbytes
        )

    def search(self, query: ArrayLike, k: int) -> list[SearchResult]:
        vector = _query_vector(query, self.dim)
        if k <= 0:
            return []
        lookup = (self._books @ vector.reshape(len(self._books), -1, 1)).squeeze(-1)
        scores = np.zeros(len(self), dtype=np.float32)
        for part, table in enumerate(lookup):
            scores += table[self._codes[:, part]]
        scores *= self._inverse_norms
        return _top_results(self._ids, scores, k)
