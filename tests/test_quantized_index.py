import numpy as np
import pytest

from cognivore.index import ProductQuantizedIndex, ScalarQuantizedIndex


def test_sq8_recovers_exact_neighbors_on_fixed_vectors() -> None:
    rng = np.random.default_rng(7)
    matrix = rng.normal(size=(600, 32)).astype(np.float32)
    queries = rng.normal(size=(12, 32)).astype(np.float32)
    labels = np.arange(100, 700)
    index = ScalarQuantizedIndex(labels, matrix)
    assert len(index) == 600
    assert index.storage_bytes == 600 * (32 + 8 + 4)

    exact = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    recovered = 0
    for query in queries:
        scores = exact @ (query / np.linalg.norm(query))
        expected = set(labels[np.argsort(scores)[-10:]])
        hits = index.search(query, 10)
        recovered += len(expected & {hit.id for hit in hits})
        for hit in hits:
            assert hit.score == pytest.approx(scores[hit.id - 100], abs=0.005)
    assert recovered >= 116


def test_pq_trains_reproducibly_on_independent_vectors() -> None:
    rng = np.random.default_rng(7)
    training = rng.normal(size=(200, 16)).astype(np.float32)
    corpus = training[:50].copy()
    corpus[0] = 0
    params = {"subspaces": 4, "clusters": 16, "seed": 71, "training_vectors": training[50:]}
    first = ProductQuantizedIndex(range(50), corpus, **params)
    second = ProductQuantizedIndex(range(50), corpus, **params)
    assert first.storage_bytes == 50 * (4 + 8 + 4) + 16 * 16 * 4
    hits = first.search(corpus[1], 50)
    assert hits == second.search(corpus[1], 50)
    assert hits[0].id == 1
    assert next(hit.score for hit in hits if hit.id == 0) == 0.0
    assert all(-1.0 <= hit.score <= 1.0 for hit in hits)
    assert first.search(corpus[1], 0) == []


def test_quantized_validation_and_empty_corpus() -> None:
    empty = ScalarQuantizedIndex([], np.empty((0, 4)))
    assert len(empty) == 0
    assert empty.search([1, 0, 0, 0], 10) == []
    with pytest.raises(ValueError, match="finite"):
        ScalarQuantizedIndex([1], [[np.nan, 1]])
    with pytest.raises(ValueError, match="integer ID"):
        ScalarQuantizedIndex([1.5], [[1, 0]])
    with pytest.raises(ValueError, match="query of length"):
        empty.search([1], 1)
    with pytest.raises(ValueError, match="subspaces"):
        ProductQuantizedIndex([1, 2], [[1, 0], [0, 1]], subspaces=3)
    with pytest.raises(ValueError, match="at least one vector per cluster"):
        ProductQuantizedIndex([1, 2], [[1, 0], [0, 1]], subspaces=2, clusters=3)
