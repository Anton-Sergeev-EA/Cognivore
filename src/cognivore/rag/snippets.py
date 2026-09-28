"""Query-focused snippets: the part of a retrieved chunk that actually
answers the question, instead of simply its first N characters.

Chunks are sized for retrieval (~800 characters), but the agent's search
tool shows the model a much shorter snippet to keep prompt prefill fast on
CPU. Cutting a chunk at a fixed offset regularly drops exactly the
sentence that matters -- especially in Chinese, where 800 characters span
several sections. Here the snippet is centred on the best-matching
sentence and grown with its neighbours up to the length limit.
"""

from __future__ import annotations

import re

from cognivore.rag.store import RetrievedChunk
from cognivore.rag.tokenize import STOPWORDS, is_cjk_run, tokenize

# Sentence terminators: ASCII . ! ?, the Devanagari danda and double danda
# (U+0964, U+0965), the CJK full stop (U+3002) and fullwidth ! ? (U+FF01,
# U+FF1F); newlines also end a sentence.
# re.M so that "$" also ends a sentence at a line break: a line without
# terminal punctuation (a heading, a wrapped line) is still a sentence.
_SENTENCE_RE = re.compile(
    r"[^.!?\u0964\u0965\u3002\uff01\uff1f\n]+(?:[.!?\u0964\u0965\u3002\uff01\uff1f]+|$)",
    re.UNICODE | re.M,
)


def split_sentences(text: str) -> list[tuple[int, int]]:
    """Sentence spans as ``(start, end)`` offsets, whitespace trimmed."""
    spans: list[tuple[int, int]] = []
    for match in _SENTENCE_RE.finditer(text):
        start, end = match.start(), match.end()
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if end > start:
            spans.append((start, end))
    return spans


def _stem(token: str) -> str:
    """Crude, language-agnostic stemming: the first five letters of longer
    alphabetic words, so inflected forms ("тарифов"/"тариф",
    "remboursement"/"rembourser") still match. CJK tokens are kept whole."""
    return token if is_cjk_run(token) or len(token) <= 5 else token[:5]


def _stems(text: str) -> set[str]:
    return {_stem(t) for t in tokenize(text) if t not in STOPWORDS}


def best_snippet(text: str, query: str, limit: int = 300) -> str:
    """The ``limit``-character window of ``text`` most relevant to
    ``query`` (falls back to the beginning when nothing matches)."""
    text = text.strip()
    if len(text) <= limit:
        return text
    spans = split_sentences(text)
    query_stems = _stems(query)
    if not spans or not query_stems:
        return text[:limit]

    def score(span: tuple[int, int]) -> int:
        return len(query_stems & _stems(text[span[0] : span[1]]))

    scores = [score(span) for span in spans]
    best = max(range(len(spans)), key=lambda i: (scores[i], -i))
    if scores[best] == 0:
        return text[:limit]

    lo = hi = best
    start, end = spans[best]
    if end - start >= limit:
        return text[start : start + limit]
    # Grow forwards: the answer usually follows the matching heading or
    # topic sentence. Only when the chunk ends too soon to give any
    # context is the window extended backwards -- otherwise the snippet
    # would open with the tail of an unrelated preceding section.
    while hi + 1 < len(spans) and spans[hi + 1][1] - start <= limit:
        hi += 1
        end = spans[hi][1]
    while end - start < limit // 3 and lo > 0 and end - spans[lo - 1][0] <= limit:
        lo -= 1
        start = spans[lo][0]
    return text[start:end]


_WS_RE = re.compile(r"\s+")


def distinct_snippets(
    hits: list[RetrievedChunk], query: str, limit: int = 300, top_k: int | None = None
) -> list[tuple[RetrievedChunk, str]]:
    """Pairs each hit with its :func:`best_snippet`, dropping hits whose
    snippet repeats one already kept.

    Neighbouring chunks overlap (``chunk_overlap``), so two of them often
    centre on the very same sentence; showing both -- to the model or in the
    UI's source list -- repeats a passage and hides a genuinely different
    one. Order (i.e. rank) is preserved; ``top_k`` caps the result.
    """
    kept: list[tuple[RetrievedChunk, str]] = []
    seen: list[str] = []
    for hit in hits:
        snippet = best_snippet(hit.text, query, limit=limit)
        key = _WS_RE.sub(" ", snippet).strip().lower()
        if any(key in other or other in key for other in seen):
            continue
        seen.append(key)
        kept.append((hit, snippet))
        if top_k is not None and len(kept) >= top_k:
            break
    return kept
