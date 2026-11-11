"""Measure source-line finding quality on the checked-in review corpus."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from difflib import SequenceMatcher
from functools import partial
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import TypedDict

import numpy as np

from cognivore.rag.embeddings import EmbeddingModel, FastEmbedModel, HashingEmbedder
from cognivore.review import compare_requirements

FindingKey = tuple[str, int | None, int | None]
Findings = set[FindingKey]


class Case(TypedDict):
    name: str
    before: list[str]
    after: list[str]
    expected: list[FindingKey]


def line_diff(before: list[str], after: list[str]) -> Findings:
    findings: Findings = set()
    for tag, start, end, first, last in SequenceMatcher(
        None, before, after, autojunk=False
    ).get_opcodes():
        if tag == "equal":
            continue
        replacements = min(end - start, last - first) if tag == "replace" else 0
        findings.update(("changed", start + i + 1, first + i + 1) for i in range(replacements))
        findings.update(("removed", i + 1, None) for i in range(start + replacements, end))
        findings.update(("added", None, j + 1) for j in range(first + replacements, last))
    return findings


def vector_only(before: list[str], after: list[str]) -> Findings:
    if not before or not after:
        return line_diff(before, after)
    vectors = np.asarray(HashingEmbedder().embed(before + after))
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors /= np.maximum(norms, 1e-8)
    scores = vectors[: len(before)] @ vectors[len(before) :].T
    pairs = sorted(np.ndindex(scores.shape), key=lambda pair: -scores[pair])
    findings: Findings = set()
    left, right = set(range(len(before))), set(range(len(after)))
    for i, j in pairs:
        if scores[i, j] < 0.50:
            break
        if i not in left or j not in right:
            continue
        left.remove(i)
        right.remove(j)
        if before[i] != after[j]:
            findings.add(("changed", i + 1, j + 1))
    findings.update(("removed", i + 1, None) for i in left)
    findings.update(("added", None, j + 1) for j in right)
    return findings


def without_headings(
    method: Callable[[list[str], list[str]], Findings], before: list[str], after: list[str]
) -> Findings:
    left, right = (
        [
            index
            for index, text in enumerate(lines, 1)
            if text.strip() and not text.lstrip().startswith("#")
        ]
        for lines in (before, after)
    )
    return {
        (kind, left[i - 1] if i is not None else None, right[j - 1] if j is not None else None)
        for kind, i, j in method([before[i - 1] for i in left], [after[j - 1] for j in right])
    }


def cognivore(
    before: list[str],
    after: list[str],
    *,
    embedder: EmbeddingModel,
    use_section_context: bool = False,
) -> Findings:
    findings = compare_requirements(
        "before",
        "\n".join(before),
        "after",
        "\n".join(after),
        embedder,
        use_section_context=use_section_context,
    )
    return {
        (
            finding.kind,
            finding.before.line if finding.before else None,
            finding.after.line if finding.after and finding.kind != "ambiguous" else None,
        )
        for finding in findings
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--cases", type=Path, default=Path(__file__).with_name("review_cases.json"))
    parser.add_argument(
        "--semantic-model", help="Also test a FastEmbed model with and without section context."
    )
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    cases: list[Case] = json.loads(args.cases.read_text(encoding="utf-8"))
    if not cases:
        parser.error("the corpus must contain at least one case")
    methods: dict[str, Callable[[list[str], list[str]], Findings]] = {
        "line diff": partial(without_headings, line_diff),
        "vector only": partial(without_headings, vector_only),
        "Cognivore (hashing)": partial(cognivore, embedder=HashingEmbedder()),
    }
    if args.semantic_model:
        semantic = FastEmbedModel(args.semantic_model)
        methods["Cognivore (semantic)"] = partial(cognivore, embedder=semantic)
        methods["Cognivore (semantic + sections)"] = partial(
            cognivore, embedder=semantic, use_section_context=True
        )
    for name, method in methods.items():
        tp = fp = fn = 0
        elapsed = []
        for case in cases:
            expected = {tuple(key) for key in case["expected"]}
            actual = method(case["before"], case["after"])
            tp += len(actual & expected)
            fp += len(actual - expected)
            fn += len(expected - actual)
            if actual != expected:
                print(
                    f"{name} | {case['name']} | extra={actual - expected} missed={expected - actual}"
                )
        for _ in range(args.repeat):
            start = perf_counter()
            for case in cases:
                method(case["before"], case["after"])
            elapsed.append((perf_counter() - start) * 1000 / len(cases))
        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        print(
            f"{name}: TP={tp} FP={fp} FN={fn} precision={precision:.1%} "
            f"recall={recall:.1%} median={median(elapsed):.3f} ms/case"
        )


if __name__ == "__main__":
    main()
