#!/usr/bin/env python3
"""Standalone benchmark runner: ``python benchmarks/bench_index.py``.

Prints a table comparing the pure-NumPy flat index, the C++ FlatIndex, and
the C++ NSWIndex (build time, search time, and the NSW index's recall@k
against the exact result). The numbers quoted in README.md come from this
script.
"""

from __future__ import annotations

import sys

from rich.console import Console

sys.path.insert(0, "src")

from cognivore.benchmark import run_benchmark, run_scaling_benchmark

if __name__ == "__main__":
    console = Console()
    run_benchmark(n=3000, dim=384, queries=300, k=10, console=console)
    run_scaling_benchmark(sizes=(1_000, 10_000, 50_000), dim=384, console=console)
