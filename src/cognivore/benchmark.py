"""Benchmarks the native C++ indexes against the pure-NumPy fallback.

Used by ``cognivore bench`` and by ``benchmarks/bench_index.py`` (a thin
standalone script for running this outside an installed package, e.g. to
paste fresh numbers into README.md).

Two separate things are measured, deliberately:

1. ``run_benchmark`` -- build/search time and recall@k at a modest size,
   comparing the pure-NumPy fallback, the exact C++ FlatIndex, and the
   approximate C++ NSWIndex (at a few `ef` settings, since recall vs. speed
   is a tunable trade-off, not a single number).
2. ``run_scaling_benchmark`` -- how per-query search latency grows with
   collection size for FlatIndex (must scan every vector: linear) vs.
   NSWIndex (graph walk: roughly logarithmic). This is where an ANN index
   is supposed to win, and a benchmark that only tests a few thousand
   vectors is too small to show it -- brute force is *fine*, and can even
   beat a naive parallel loop against a BLAS-backed NumPy matmul, until the
   collection is large enough that scanning it becomes the bottleneck.

Random, uniformly-distributed high-dimensional vectors (what both
benchmarks generate) are close to a worst case for ANN recall -- with no
real cluster structure, "nearest" neighbours are only marginally closer
than random ones, so small search-order perturbations change the top-k
easily. Real embeddings (semantically clustered) recall noticeably better
at the same `ef`; that's noted in README.md alongside these numbers rather
than hidden by cherry-picking an easier synthetic distribution.
"""

from __future__ import annotations

import time

import numpy as np

from cognivore.index import is_native
from cognivore.index.python_index import FlatIndexPy


def _build_and_time(index, vectors: np.ndarray) -> float:
    start = time.perf_counter()
    for i, vec in enumerate(vectors):
        index.add(i, vec.tolist())
    return time.perf_counter() - start


def _search_and_time(index, queries: np.ndarray, k: int, ef: int | None = None) -> float:
    start = time.perf_counter()
    for q in queries:
        if ef is not None:
            index.search(q.tolist(), k, ef)
        else:
            index.search(q.tolist(), k)
    return time.perf_counter() - start


def _recall_at(flat, nsw, queries: np.ndarray, k: int, ef: int) -> float:
    total = 0.0
    for q in queries:
        exact = {r.id for r in flat.search(q.tolist(), k)}
        approx = {r.id for r in nsw.search(q.tolist(), k, ef)}
        total += len(exact & approx) / k
    return total / len(queries)


def run_benchmark(
    n: int = 3000,
    dim: int = 384,
    queries: int = 200,
    k: int = 10,
    ef_sweep: tuple[int, ...] = (50, 150, 400),
    console=None,
) -> dict:
    rng = np.random.default_rng(42)
    vectors = rng.normal(size=(n, dim)).astype(np.float32)
    query_vectors = rng.normal(size=(queries, dim)).astype(np.float32)

    results: dict[str, dict[str, float]] = {}

    py_flat = FlatIndexPy(dim)
    results["python_flat (NumPy)"] = {
        "build_s": _build_and_time(py_flat, vectors),
        "search_s": _search_and_time(py_flat, query_vectors, k),
    }

    if is_native:
        from cognivore import _native

        cpp_flat = _native.FlatIndex(dim)
        results["cpp_flat (exact)"] = {
            "build_s": _build_and_time(cpp_flat, vectors),
            "search_s": _search_and_time(cpp_flat, query_vectors, k),
        }

        cpp_nsw = _native.NSWIndex(dim, 16, 200)
        build_s = _build_and_time(cpp_nsw, vectors)
        for ef in ef_sweep:
            search_s = _search_and_time(cpp_nsw, query_vectors, k, ef=ef)
            recall = _recall_at(cpp_flat, cpp_nsw, query_vectors, k, ef)
            results[f"cpp_nsw (ef={ef})"] = {
                "build_s": build_s,
                "search_s": search_s,
                "recall_at_k": recall,
            }

    if console is not None:
        _print_table(console, results, n, dim, queries, k)
    return results


def run_scaling_benchmark(
    sizes: tuple[int, ...] = (1_000, 10_000, 50_000),
    dim: int = 384,
    queries: int = 100,
    k: int = 10,
    ef: int = 150,
    console=None,
) -> dict:
    """Shows how per-query latency scales with collection size for the
    exact (linear-scan) index vs. the approximate graph index. Skips the
    pure-NumPy fallback here since its O(n) `add()` (see FlatIndexPy's
    docstring) makes building the larger sizes impractically slow -- this
    benchmark is about *search* scaling, not the fallback's build cost.
    """
    if not is_native:
        if console is not None:
            console.print(
                "[yellow]Native extension not built; skipping scaling benchmark.[/yellow]"
            )
        return {}

    from cognivore import _native

    rng = np.random.default_rng(7)
    results: dict[int, dict[str, float]] = {}
    for n in sizes:
        vectors = rng.normal(size=(n, dim)).astype(np.float32)
        query_vectors = rng.normal(size=(queries, dim)).astype(np.float32)

        flat = _native.FlatIndex(dim)
        for i, v in enumerate(vectors):
            flat.add(i, v.tolist())
        flat_ms = _search_and_time(flat, query_vectors, k) / queries * 1000

        nsw = _native.NSWIndex(dim, 16, 200)
        for i, v in enumerate(vectors):
            nsw.add(i, v.tolist())
        nsw_ms = _search_and_time(nsw, query_vectors, k, ef=ef) / queries * 1000

        results[n] = {"flat_ms_per_query": flat_ms, "nsw_ms_per_query": nsw_ms}

    if console is not None:
        from rich.table import Table

        table = Table(title=f"Search latency vs. collection size (dim={dim}, ef={ef})")
        table.add_column("n")
        table.add_column("FlatIndex ms/query", justify="right")
        table.add_column("NSWIndex ms/query", justify="right")
        table.add_column("Speedup", justify="right")
        for n, stats in results.items():
            speedup = stats["flat_ms_per_query"] / max(stats["nsw_ms_per_query"], 1e-9)
            table.add_row(
                str(n),
                f"{stats['flat_ms_per_query']:.3f}",
                f"{stats['nsw_ms_per_query']:.3f}",
                f"{speedup:.1f}x",
            )
        console.print(table)
    return results


def _print_table(console, results: dict, n: int, dim: int, queries: int, k: int) -> None:
    from rich.table import Table

    table = Table(title=f"Index benchmark (n={n}, dim={dim}, {queries} queries, k={k})")
    table.add_column("Index")
    table.add_column("Build (s)", justify="right")
    table.add_column("Search / query (ms)", justify="right")
    table.add_column("Recall@k", justify="right")
    for name, stats in results.items():
        table.add_row(
            name,
            f"{stats['build_s']:.3f}",
            f"{stats['search_s'] / queries * 1000:.3f}",
            f"{stats['recall_at_k']:.3f}" if "recall_at_k" in stats else "1.000 (exact)",
        )
    console.print(table)
