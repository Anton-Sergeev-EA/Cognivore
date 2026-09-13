"""Correctness tests for the vector index backends.

If the native C++ extension isn't built, the native-only tests are skipped
(``is_native`` is False) rather than failing, so the suite stays green on a
machine with no C++ toolchain -- the fallback path is exactly what
:mod:`cognivore.index` degrades to in that case, and is covered by the
`test_flat_index_py_*` tests regardless.
"""

from __future__ import annotations

import numpy as np
import pytest

from cognivore.index import is_native
from cognivore.index.python_index import FlatIndexPy

pytestmark_native = pytest.mark.skipif(not is_native, reason="native C++ extension not built")


def _random_vectors(n: int, dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, dim)).astype(np.float32)


def test_flat_index_py_returns_exact_top_k() -> None:
    idx = FlatIndexPy(dim=8)
    vectors = _random_vectors(50, 8)
    for i, v in enumerate(vectors):
        idx.add(i, v.tolist())

    query = vectors[7]  # searching for a vector already in the index
    results = idx.search(query.tolist(), k=3)
    assert results[0].id == 7
    assert results[0].score == pytest.approx(1.0, abs=1e-4)


def test_flat_index_py_serialize_roundtrip() -> None:
    idx = FlatIndexPy(dim=4)
    for i, v in enumerate([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]):
        idx.add(i, v)
    blob = idx.serialize()
    restored = FlatIndexPy.deserialize(blob)
    assert len(restored) == len(idx)
    assert restored.search([1, 0, 0, 0], k=1)[0].id == 0


@pytestmark_native
def test_native_flat_index_matches_python_flat_index() -> None:
    from cognivore import _native

    dim = 16
    vectors = _random_vectors(100, dim, seed=1)
    query = _random_vectors(1, dim, seed=2)[0]

    cpp_idx = _native.FlatIndex(dim)
    py_idx = FlatIndexPy(dim)
    for i, v in enumerate(vectors):
        cpp_idx.add(i, v.tolist())
        py_idx.add(i, v.tolist())

    cpp_results = cpp_idx.search(query.tolist(), 5)
    py_results = py_idx.search(query.tolist(), 5)

    assert [r.id for r in cpp_results] == [r.id for r in py_results]
    for cpp_r, py_r in zip(cpp_results, py_results, strict=True):
        assert cpp_r.score == pytest.approx(py_r.score, abs=1e-4)


@pytestmark_native
def test_native_flat_index_serialize_roundtrip() -> None:
    from cognivore import _native

    dim = 8
    idx = _native.FlatIndex(dim)
    for i, v in enumerate(_random_vectors(20, dim)):
        idx.add(i, v.tolist())
    blob = idx.serialize()
    assert isinstance(blob, bytes)
    restored = _native.FlatIndex.deserialize(blob)
    assert len(restored) == len(idx)

    query = _random_vectors(1, dim, seed=99)[0].tolist()
    assert [r.id for r in idx.search(query, 5)] == [r.id for r in restored.search(query, 5)]


@pytestmark_native
def test_native_nsw_index_recall_is_reasonably_high() -> None:
    from cognivore import _native

    dim = 32
    n = 800
    vectors = _random_vectors(n, dim, seed=3)

    flat = _native.FlatIndex(dim)
    nsw = _native.NSWIndex(dim, 16, 100)
    for i, v in enumerate(vectors):
        flat.add(i, v.tolist())
        nsw.add(i, v.tolist())

    queries = _random_vectors(30, dim, seed=4)
    total_recall = 0.0
    k = 10
    for q in queries:
        exact_ids = {r.id for r in flat.search(q.tolist(), k)}
        approx_ids = {r.id for r in nsw.search(q.tolist(), k, 80)}
        total_recall += len(exact_ids & approx_ids) / k
    mean_recall = total_recall / len(queries)

    # A single-layer NSW with these parameters should comfortably recover
    # most of the true top-k on random Gaussian data; this is a regression
    # guard, not a tight bound.
    assert mean_recall >= 0.8, f"NSW recall dropped to {mean_recall:.2f}"


@pytestmark_native
def test_native_nsw_index_serialize_roundtrip_preserves_search() -> None:
    from cognivore import _native

    dim = 12
    idx = _native.NSWIndex(dim, 8, 50)
    for i, v in enumerate(_random_vectors(60, dim, seed=5)):
        idx.add(i, v.tolist())

    query = _random_vectors(1, dim, seed=6)[0].tolist()
    before = [r.id for r in idx.search(query, 5, 50)]

    restored = _native.NSWIndex.deserialize(idx.serialize())
    after = [r.id for r in restored.search(query, 5, 50)]
    assert before == after


@pytestmark_native
def test_native_index_rejects_wrong_dimension() -> None:
    from cognivore import _native

    idx = _native.FlatIndex(4)
    idx.add(0, [1.0, 2.0, 3.0, 4.0])
    with pytest.raises(Exception):  # noqa: B017 - pybind11 surfaces std::invalid_argument as RuntimeError
        idx.add(1, [1.0, 2.0, 3.0])  # wrong length
