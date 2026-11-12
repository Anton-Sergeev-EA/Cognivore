"""Pure NumPy fallback for when the compiled ``cognivore._native`` extension
isn't available (e.g. no C++ toolchain at install time).

Only exact (flat) search is provided here -- the approximate NSW graph
index is C++-only, since its entire point is to demonstrate a hand-written
ANN data structure. Everything in :mod:`cognivore.rag` and
:mod:`cognivore.agent` is written against the small common protocol in
``cognivore.index.__init__`` so callers never need to know which backend is
active.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

_HEADER = struct.Struct("<QQ")  # dim, count


@dataclass(frozen=True)
class SearchResult:
    id: int
    score: float


class FlatIndexPy:
    """Exact brute-force cosine-similarity search, pure NumPy.

    Behaviourally equivalent to the C++ ``FlatIndex`` (same normalization,
    same tie-breaking via stable sort), just slower on large collections --
    see ``benchmarks/bench_index.py`` for a head-to-head comparison.
    """

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._ids: list[int] = []
        self._vectors: NDArray[np.float32] = np.empty((0, dim), dtype=np.float32)

    def add(self, id: int, vector: list[float]) -> None:
        vec = np.asarray(vector, dtype=np.float32)
        if vec.shape != (self.dim,):
            raise ValueError(f"expected a vector of length {self.dim}, got {vec.shape}")
        norm = np.linalg.norm(vec)
        if norm > 1e-10:
            vec = vec / norm
        self._ids.append(id)
        self._vectors = np.vstack([self._vectors, vec[None, :]])

    @classmethod
    def from_vectors(cls, dim: int, ids: list[int], matrix: NDArray[np.float32]) -> FlatIndexPy:
        """Build in one pass when restoring a store from persisted raw vectors."""
        if matrix.shape != (len(ids), dim):
            raise ValueError("vector matrix shape does not match IDs and dimensions")
        index = cls(dim)
        index._ids = list(ids)
        index._vectors = np.array(matrix, dtype=np.float32, copy=True)
        norms = np.linalg.norm(index._vectors, axis=1, keepdims=True)
        np.divide(index._vectors, norms, out=index._vectors, where=norms > 1e-10)
        return index

    def search(self, query: list[float], k: int) -> list[SearchResult]:
        if not self._ids:
            return []
        q = np.asarray(query, dtype=np.float32)
        norm = np.linalg.norm(q)
        if norm > 1e-10:
            q = q / norm
        scores = self._vectors @ q
        k = min(k, len(self._ids))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [SearchResult(id=self._ids[i], score=float(scores[i])) for i in top]

    def __len__(self) -> int:
        return len(self._ids)

    def serialize(self) -> bytes:
        """Binary layout matches the C++ FlatIndex format (little-endian
        u64 dim, u64 count, int64 ids[], float32 vectors[]) so the same
        file format works whether or not the native extension is built."""
        ids_bytes = np.asarray(self._ids, dtype="<i8").tobytes()
        vec_bytes = self._vectors.astype("<f4").tobytes()
        return _HEADER.pack(self.dim, len(self._ids)) + ids_bytes + vec_bytes

    @classmethod
    def deserialize(cls, blob: bytes) -> FlatIndexPy:
        dim, count = _HEADER.unpack_from(blob, 0)
        offset = _HEADER.size
        ids = np.frombuffer(blob, dtype="<i8", count=count, offset=offset).tolist()
        offset += count * 8
        vectors = np.frombuffer(blob, dtype="<f4", count=count * dim, offset=offset).reshape(
            count, dim
        )
        idx = cls(dim)
        idx._ids = ids
        idx._vectors = vectors.astype(np.float32).copy()
        return idx
