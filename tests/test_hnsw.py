from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from cognivore.index import HNSWIndex

pytestmark = pytest.mark.skipif(HNSWIndex is None, reason="native HNSW unavailable")


def test_hnsw_search_uses_multiple_levels_and_recovers_neighbors() -> None:
    rng = np.random.default_rng(123)
    vectors = rng.normal(size=(1200, 32)).astype(np.float32)
    queries = rng.normal(size=(20, 32)).astype(np.float32)
    index = HNSWIndex(32, m=12, ef_construction=100, seed=42)
    for key, vector in enumerate(vectors, 100):
        index.add(key, vector)
    assert len(index) == len(vectors)
    assert index.max_level >= 2

    normalized = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    recovered = 0
    for query in queries:
        scores = normalized @ (query / np.linalg.norm(query))
        exact = set(np.argsort(scores)[-10:] + 100)
        hits = index.search(query, 10, ef=90)
        assert len(hits) == 10
        assert [hit.score for hit in hits] == sorted((hit.score for hit in hits), reverse=True)
        recovered += len(exact & {hit.id for hit in hits})
        for hit in hits:
            assert hit.score == pytest.approx(scores[hit.id - 100], abs=1e-6)
    assert recovered / (len(queries) * 10) >= 0.95


def test_hnsw_boundary_inputs() -> None:
    with pytest.raises(ValueError, match="dimension"):
        HNSWIndex(0)
    with pytest.raises(ValueError, match="ef_construction"):
        HNSWIndex(2, m=16, ef_construction=8)
    index = HNSWIndex(2)
    assert index.search([1, 0], 1) == []
    with pytest.raises(ValueError, match="finite"):
        index.add(1, [float("nan"), 0])
    with pytest.raises(ValueError, match="one-dimensional"):
        index.add(1, [[1, 0]])
    assert len(index) == 0
    index.add(-1, [0, 0])
    index.add(10, [1e30, 0])
    assert index.search([1, 0], 0) == []
    assert [hit.id for hit in index.search([1, 0], 4, ef=0)] == [10, -1]


def test_hnsw_allows_concurrent_insertion_and_search() -> None:
    index = HNSWIndex(2)
    index.add(-1, [1, 0])

    def insert_or_query(key: int) -> None:
        if key % 2 == 0:
            index.add(key, [0, 1])
        else:
            assert index.search([1, 0], 1)[0].id == -1

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(insert_or_query, range(80)))
    assert len(index) == 41
