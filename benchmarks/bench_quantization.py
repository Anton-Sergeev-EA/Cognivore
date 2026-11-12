"""Compare compressed scans with an exact float32 cosine oracle."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from time import perf_counter

import numpy as np
from numpy.typing import NDArray

from cognivore.index import ProductQuantizedIndex, ScalarQuantizedIndex
from cognivore.index.python_index import SearchResult

K = 10


def _normalized(rows: NDArray[np.float32]) -> NDArray[np.float32]:
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


def _evaluate(
    label: str,
    search: Callable[[NDArray[np.float32]], list[SearchResult]],
    queries: NDArray[np.float32],
    truth: NDArray[np.intp],
    build_seconds: float,
    storage: int,
) -> None:
    search(queries[0])
    latencies = []
    recovered = 0
    for query, exact in zip(queries, truth, strict=True):
        start = perf_counter()
        hits = search(query)
        latencies.append((perf_counter() - start) * 1_000)
        recovered += len(set(exact) & {hit.id for hit in hits})
    print(
        f"| {label} | {storage:,} | {build_seconds:.3f} | "
        f"{recovered / truth.size:.3f} | {np.median(latencies):.3f} | "
        f"{np.percentile(latencies, 95):.3f} |"
    )


def run(size: int, dim: int, n_queries: int, distribution: str) -> None:
    rng = np.random.default_rng(42)
    centers = rng.standard_normal((64, dim), dtype=np.float32)

    def sample(count: int) -> NDArray[np.float32]:
        noise = rng.standard_normal((count, dim), dtype=np.float32)
        if distribution == "clustered":
            noise *= 0.25
            noise += centers[rng.integers(0, len(centers), count)]
        return _normalized(noise)

    vectors, queries, training = sample(size), sample(n_queries), sample(4_096)
    ids = np.arange(size, dtype=np.int64)
    truth = np.stack([np.argpartition(-(vectors @ query), K - 1)[:K] for query in queries])
    start = perf_counter()
    exact_vectors = vectors.copy()
    exact_ids = ids.copy()
    exact_build = perf_counter() - start

    def rank(query: NDArray[np.float32], candidates: NDArray[np.int64]) -> list[SearchResult]:
        scores = exact_vectors[candidates] @ query
        best = np.argsort(-scores, kind="stable")[:K]
        return [SearchResult(int(exact_ids[candidates[i]]), float(scores[i])) for i in best]

    def exact(query: NDArray[np.float32]) -> list[SearchResult]:
        scores = exact_vectors @ query
        candidates = np.argpartition(-scores, K - 1)[:K]
        best = candidates[np.argsort(-scores[candidates])]
        return [SearchResult(int(exact_ids[i]), float(scores[i])) for i in best]

    print(f"\n{size:,} vectors, {dim} dimensions, {distribution}, {n_queries} queries")
    print("| Method | Retained bytes | Build (s) | Recall@10 | p50 (ms) | p95 (ms) |")
    print("|---|---:|---:|---:|---:|---:|")
    exact_bytes = exact_vectors.nbytes + exact_ids.nbytes
    _evaluate("Float32", exact, queries, truth, exact_build, exact_bytes)

    start = perf_counter()
    sq = ScalarQuantizedIndex(ids, vectors)
    sq_build = perf_counter() - start
    _evaluate("SQ8", lambda query: sq.search(query, K), queries, truth, sq_build, sq.storage_bytes)

    start = perf_counter()
    pq = ProductQuantizedIndex(
        ids, vectors, subspaces=dim // 8, clusters=256, iterations=12, training_vectors=training
    )
    pq_build = perf_counter() - start
    _evaluate("PQ", lambda query: pq.search(query, K), queries, truth, pq_build, pq.storage_bytes)

    def rerank(query: NDArray[np.float32]) -> list[SearchResult]:
        candidates = np.array([hit.id for hit in pq.search(query, 4 * K)], dtype=np.int64)
        return rank(query, candidates)

    _evaluate("PQ + exact rerank", rerank, queries, truth, pq_build, pq.storage_bytes + exact_bytes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[1_000, 10_000, 50_000])
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--distribution", choices=["uniform", "clustered"], default="uniform")
    args = parser.parse_args()
    if min(args.sizes) < K or args.dim < 8 or args.dim % 8 or args.queries < 1:
        parser.error(
            "sizes must be >= 10, dimension a positive multiple of 8, and queries positive"
        )
    for size in args.sizes:
        run(size, args.dim, args.queries, args.distribution)


if __name__ == "__main__":
    main()
