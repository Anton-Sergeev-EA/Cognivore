"""Names clusters with class-based TF-IDF (c-TF-IDF, as in BERTopic):
treat all text of one cluster as a single document and score each term by
how frequent it is *in that cluster* relative to how common it is overall.
"""

from __future__ import annotations

import math
from collections import Counter

from cognivore.rag.tokenize import content_tokens, is_cjk_run


def cluster_keywords(texts: list[str], labels: list[int], top_n: int = 3) -> dict[int, list[str]]:
    per_cluster: dict[int, Counter[str]] = {}
    for text, label in zip(texts, labels, strict=True):
        tokens = [t for t in content_tokens(text) if not t.isdigit()]
        per_cluster.setdefault(label, Counter()).update(tokens)

    overall: Counter[str] = Counter()
    for counts in per_cluster.values():
        overall.update(counts)
    if not per_cluster:
        return {}
    avg_words = sum(overall.values()) / len(per_cluster)
    # A word found in every cluster names none of them: boilerplate such as
    # "On screen" in video timelines, or the company name in a handbook.
    clusters_with = Counter(t for counts in per_cluster.values() for t in counts)
    everywhere = (
        {t for t, n in clusters_with.items() if n == len(per_cluster)}
        if len(per_cluster) > 1
        else set()
    )

    keywords: dict[int, list[str]] = {}
    for label, counts in per_cluster.items():
        total = sum(counts.values()) or 1
        scored = sorted(
            counts,
            key=lambda t: (-(counts[t] / total) * math.log(1 + avg_words / overall[t]), t),
        )
        chosen: list[str] = []
        used_chars: set[str] = set()
        for term in sorted(scored, key=lambda t: t in everywhere):
            # Overlapping CJK bigrams ("退款", "款政") would otherwise
            # fill the label with near-duplicates of one word.
            if is_cjk_run(term):
                if used_chars & set(term):
                    continue
                used_chars |= set(term)
            chosen.append(term)
            if len(chosen) == top_n:
                break
        keywords[label] = chosen
    return keywords
