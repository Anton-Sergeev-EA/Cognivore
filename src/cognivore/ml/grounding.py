"""Answer grounding: how much of an answer is actually supported by the
passages retrieved for it.

Each sentence of the answer is scored against every retrieved passage with
a blend of

* **lexical support** -- the share of the sentence's content words that
  appear in the passage (precise, catches numbers and names), and
* **semantic support** -- cosine similarity of sentence and passage
  embeddings (tolerates paraphrase; cross-lingual with a multilingual
  embedder).

A sentence's support is its best passage's score; the answer's score is the
average over sentences weighted by their length. Sentences with low support
are exactly the ones a reader should double-check -- the UI underlines
them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cognivore.rag.embeddings import EmbeddingModel
from cognivore.rag.snippets import split_sentences
from cognivore.rag.tokenize import content_tokens

LEXICAL_WEIGHT = 0.6
HIGH_THRESHOLD = 0.6
MEDIUM_THRESHOLD = 0.35


@dataclass
class SentenceSupport:
    start: int  # character offsets into the answer
    end: int
    support: float
    source_rank: int | None  # 1-based rank of the best-supporting passage


@dataclass
class GroundingReport:
    score: float
    level: str  # "high" | "medium" | "low"
    sentences: list[SentenceSupport] = field(default_factory=list)


def level_for(score: float) -> str:
    if score >= HIGH_THRESHOLD:
        return "high"
    if score >= MEDIUM_THRESHOLD:
        return "medium"
    return "low"


def _normalized(vectors: list[list[float]]) -> np.ndarray:
    m = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(norms > 1e-12, norms, 1.0)


def assess_grounding(
    answer: str, passages: list[str], embedder: EmbeddingModel
) -> GroundingReport | None:
    """Returns ``None`` when there is nothing to assess (no passages were
    retrieved, or the answer has no content words) -- e.g. a calculator
    answer isn't "ungrounded", grounding simply doesn't apply to it."""
    if not passages or not answer.strip():
        return None
    spans = split_sentences(answer)
    scored_spans = [(s, e, content_tokens(answer[s:e])) for s, e in spans]
    scored_spans = [(s, e, toks) for s, e, toks in scored_spans if toks]
    if not scored_spans:
        return None

    passage_tokens = [set(content_tokens(p)) for p in passages]
    passage_vecs = _normalized(embedder.embed(passages))
    sentence_vecs = _normalized(embedder.embed([answer[s:e] for s, e, _ in scored_spans]))
    cosines = np.clip(sentence_vecs @ passage_vecs.T, 0.0, 1.0)

    results: list[SentenceSupport] = []
    weighted = 0.0
    weight_total = 0
    for row, (start, end, toks) in enumerate(scored_spans):
        unique = set(toks)
        best_score, best_rank = 0.0, None
        for col, ptoks in enumerate(passage_tokens):
            lexical = len(unique & ptoks) / len(unique)
            score = LEXICAL_WEIGHT * lexical + (1 - LEXICAL_WEIGHT) * float(cosines[row, col])
            if score > best_score:
                best_score, best_rank = score, col + 1
        results.append(SentenceSupport(start, end, round(best_score, 4), best_rank))
        weighted += best_score * len(toks)
        weight_total += len(toks)

    overall = weighted / weight_total if weight_total else 0.0
    return GroundingReport(score=round(overall, 4), level=level_for(overall), sentences=results)
