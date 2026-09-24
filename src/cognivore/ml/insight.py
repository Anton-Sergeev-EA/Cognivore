"""Per-turn insight: what was retrieved, where the question sits on the
knowledge map, how grounded the answer is, and whether the question
exposed a gap in the knowledge base."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from cognivore.ml.gaps import KnowledgeGapTracker
from cognivore.ml.grounding import GroundingReport, assess_grounding
from cognivore.ml.knowledge_map import KnowledgeMapBuilder, preview
from cognivore.ml.lang import detect_language
from cognivore.rag.snippets import best_snippet
from cognivore.rag.store import DocumentStore, RetrievedChunk
from cognivore.rag.tokenize import STOPWORDS, content_tokens, is_cjk_run, tokenize

SEARCH_TOOL = "search_knowledge_base"
_CJK_RUN_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")


@dataclass
class InsightHit:
    rank: int
    id: int
    source: str
    preview: str
    score: float
    vector_score: float
    lexical_score: float


@dataclass
class TurnInsight:
    language: str
    used_knowledge_base: bool
    queries: list[str] = field(default_factory=list)
    hits: list[InsightHit] = field(default_factory=list)
    query_point: tuple[float, float] | None = None
    confidence: float = 0.0
    grounding: GroundingReport | None = None
    gap: bool = False


def query_coverage(query: str, passage: str) -> float:
    """Share of the question's meaningful content found in ``passage``.

    Latin/Cyrillic text is measured in words. Chinese/Japanese text has no
    word boundaries, and its overlapping character bigrams include
    boundary-spanning noise ("成长版多少钱" -> "长版", "版多", "少钱") that no
    document will ever contain -- so it is measured in *characters*
    instead: a character counts as found if any query bigram containing it
    occurs in the passage. Stopwords (and the characters of stopword
    bigrams such as "多少") are excluded from both counts.
    """
    passage_tokens = set(tokenize(passage))
    total = 0
    found = 0
    for token in content_tokens(query):
        if not is_cjk_run(token):
            total += 1
            found += token in passage_tokens
    for run in _CJK_RUN_RE.findall(query.lower()):
        stop = [ch in STOPWORDS for ch in run]
        for i in range(len(run) - 1):
            if run[i : i + 2] in STOPWORDS:
                stop[i] = stop[i + 1] = True
        for i, ch in enumerate(run):
            if stop[i]:
                continue
            total += 1
            if len(run) == 1:
                found += ch in passage_tokens
            else:
                left = i > 0 and run[i - 1 : i + 1] in passage_tokens
                right = i < len(run) - 1 and run[i : i + 2] in passage_tokens
                found += bool(left or right)
    return found / total if total else 0.0


def retrieval_confidence(query: str, hits: list[RetrievedChunk]) -> float:
    """How well the best hit covers the question, in ``[0, 1]``: half the
    semantic similarity, half :func:`query_coverage`. Relative BM25 scores
    can't be used here (they're normalized per query, so the best hit
    always scores 1.0 even when it's irrelevant)."""
    best = 0.0
    for hit in hits:
        semantic = min(max(hit.vector_score, 0.0), 1.0)
        best = max(best, 0.5 * semantic + 0.5 * query_coverage(query, hit.text))
    return round(best, 4)


def build_turn_insight(
    store: DocumentStore,
    question: str,
    answer: str,
    tool_queries: list[str],
    used_knowledge_base: bool,
    map_builder: KnowledgeMapBuilder | None = None,
    gap_tracker: KnowledgeGapTracker | None = None,
    top_k: int = 5,
) -> TurnInsight:
    """``tool_queries`` are the queries the agent itself sent to the search
    tool; the user's own wording is always searched too, so the map lights
    up the same passages the agent could have seen."""
    language = detect_language(question)
    insight = TurnInsight(language=language, used_knowledge_base=used_knowledge_base)
    if len(store) == 0 or not question.strip():
        return insight

    queries = list(dict.fromkeys(q for q in [*tool_queries, question] if q.strip()))
    hits = store.search_multi(queries, top_k=top_k)
    insight.queries = queries
    insight.hits = [
        InsightHit(
            rank=i,
            id=h.id,
            source=h.source,
            preview=preview(best_snippet(h.text, question, limit=260), 220),
            score=round(h.score, 4),
            vector_score=round(h.vector_score, 4),
            lexical_score=round(h.lexical_score, 4),
        )
        for i, h in enumerate(hits, start=1)
    ]
    insight.confidence = retrieval_confidence(question, hits)

    if map_builder is not None:
        kmap = map_builder.get()
        insight.query_point = kmap.project(store.embed_query(question))

    if used_knowledge_base:
        insight.grounding = assess_grounding(answer, [h.text for h in hits], store.embedder)
        if gap_tracker is not None:
            insight.gap = gap_tracker.observe(question, insight.confidence, language)
    return insight
