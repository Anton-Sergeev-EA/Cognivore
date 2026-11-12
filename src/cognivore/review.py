"""Evidence-backed matching for UTF-8 specifications."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Literal

import numpy as np
from pydantic import BaseModel

from cognivore.rag.embeddings import EmbeddingModel, HashingEmbedder
from cognivore.rag.tokenize import tokenize

MAX_DOCUMENT_BYTES = 128 * 1024
MAX_LINES = 250

_ID = re.compile(r"^\s*(?:[-*]\s+)?\[?([A-Z][A-Z0-9]*-\d+)\]?[:.\s]")
_NUMBER = re.compile(
    r"(?<!\w)[+-]?\d+(?:[.,]\d+)?(?:\s*(?:%|ms|s|sec(?:onds?)?|min(?:utes?)?|"
    r"h(?:ours?)?|days?|minuten?|sekunden?|tage[n]?|мин(?:ут[уы]?)?|сек(?:унд[уы]?)?|дн(?:ей|я)|"
    r"v|kv|kw|w|mb|gb)(?!\w))?",
    re.IGNORECASE,
)
_MODAL = re.compile(
    r"\b(?:must not|shall not|may not|should not|must|shall|may|should|required|optional|"
    r"darf nicht|soll nicht|muss|müssen|darf|soll|"
    r"не должен|не должна|не допускается|должен|должна|должно|может|обязательно)\b",
    re.IGNORECASE,
)
_CONDITION = re.compile(
    r"\b(?:unless|except(?: when| if)?|only if|when|if|provided that|during|after|before|"
    r"nur wenn|wenn|falls|sofern|außer|während|nach|vor|"
    r"только если|если|когда|кроме|при условии|после|перед)\b.*",
    re.IGNORECASE,
)
_ALIASES = {
    "retain": "keep",
    "preserve": "keep",
    "backup": "reserve",
    "minimum": "least",
    "maximum": "most",
    "records": "logs",
    "shall": "must",
}
_STOP = {"a", "an", "and", "for", "of", "the", "to", "is", "must", "may", "should"}


class Citation(BaseModel):
    source: str
    line: int
    text: str
    section: str = ""
    page: int | None = None
    paragraph: int | None = None


class Finding(BaseModel):
    kind: Literal["changed", "added", "removed", "ambiguous"]
    before: Citation | None = None
    after: Citation | None = None
    signals: list[str] = []
    alternatives: list[Citation] = []


def text_citations(source: str, text: str) -> list[Citation]:
    if len(text.encode("utf-8")) > MAX_DOCUMENT_BYTES:
        raise ValueError("Each document must be at most 128 KiB.")
    citations = []
    section = ""
    for line, value in enumerate(text.splitlines(), 1):
        stripped = value.strip()
        if stripped.startswith("#"):
            section = stripped.strip("# ").strip()
        elif stripped:
            citations.append(Citation(source=source, line=line, text=value, section=section))
    if len(citations) > MAX_LINES:
        raise ValueError(f"Each document must have at most {MAX_LINES} non-heading lines.")
    return citations


def _requirement_id(citation: Citation) -> str | None:
    match = _ID.match(citation.text)
    return match[1] if match else None


def _matching_text(citation: Citation) -> str:
    return _NUMBER.sub("", _ID.sub("", citation.text))


def _terms(text: str) -> set[str]:
    return {_ALIASES.get(word, word) for word in tokenize(text) if word not in _STOP}


def _signals(before: str, after: str) -> list[str]:
    before, after = _ID.sub("", before), _ID.sub("", after)
    return [
        label
        for label, pattern in (
            ("number or unit", _NUMBER),
            ("obligation", _MODAL),
            ("condition or exception", _CONDITION),
        )
        if [value.casefold() for value in pattern.findall(before)]
        != [value.casefold() for value in pattern.findall(after)]
    ] or ["wording"]


def _similarities(embedder: EmbeddingModel, texts: list[str], split: int) -> np.ndarray:
    vectors = np.asarray(embedder.embed(texts), dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors /= np.maximum(norms, 1e-8)
    return vectors[:split] @ vectors[split:].T


def compare_requirements(
    before_name: str,
    before_text: str,
    after_name: str,
    after_text: str,
    embedder: EmbeddingModel,
    *,
    use_section_context: bool = True,
) -> list[Finding]:
    return compare_citations(
        text_citations(before_name, before_text),
        text_citations(after_name, after_text),
        embedder,
        use_section_context=use_section_context,
    )


def compare_citations(
    before: list[Citation],
    after: list[Citation],
    embedder: EmbeddingModel,
    *,
    use_section_context: bool = True,
) -> list[Finding]:
    unmatched = set(range(len(after)))
    paired: dict[int, int] = {}
    by_text: dict[str, list[int]] = defaultdict(list)
    by_id: dict[str, list[int]] = defaultdict(list)
    for j, citation in enumerate(after):
        by_text[citation.text.strip()].append(j)
        if identifier := _requirement_id(citation):
            by_id[identifier].append(j)

    # Preserve duplicates and match exact lines before considering moved or edited text.
    for i, citation in enumerate(before):
        exact = next((j for j in by_text[citation.text.strip()] if j in unmatched), None)
        if exact is not None:
            paired[i] = exact
            unmatched.remove(exact)
    before_ids = Counter(map(_requirement_id, before))
    for i, citation in enumerate(before):
        identifier = _requirement_id(citation)
        candidates = by_id.get(identifier, []) if identifier else []
        if (
            i not in paired
            and before_ids[identifier] == 1
            and len(candidates) == 1
            and candidates[0] in unmatched
        ):
            paired[i] = candidates[0]
            unmatched.remove(candidates[0])

    left = [i for i in range(len(before)) if i not in paired]
    right = sorted(unmatched)
    uncertain: dict[int, list[int]] = {}
    if left and right:
        passages = [before[i] for i in left] + [after[j] for j in right]
        texts = [_matching_text(citation) for citation in passages]
        semantic = _similarities(embedder, texts, len(left))
        contextual = None
        if (
            use_section_context
            and not isinstance(embedder, HashingEmbedder)
            and any(before[i].section for i in left)
            and any(after[j].section for j in right)
        ):
            context_texts = [
                f"{citation.section}\n{text}" if citation.section else text
                for citation, text in zip(passages, texts, strict=True)
            ]
            contextual = _similarities(embedder, context_texts, len(left))
        terms = [_terms(value) for value in texts]
        scores = np.zeros_like(semantic)
        renumbered: set[tuple[int, int]] = set()
        for x, i in enumerate(left):
            old_id = _requirement_id(before[i])
            for y, j in enumerate(right):
                new_id = _requirement_id(after[j])
                first, second = terms[x], terms[len(left) + y]
                if old_id and new_id and old_id != new_id:
                    if first and texts[x].strip() == texts[len(left) + y].strip():
                        scores[x, y] = 1.0
                        renumbered.add((x, y))
                    continue
                overlap = len(first & second) / max(1, len(first | second))
                similarity = max(0, semantic[x, y])
                score = 0.55 * overlap + 0.45 * similarity
                if (
                    contextual is not None
                    and before[i].section
                    and after[j].section
                    and similarity >= 0.55
                    and contextual[x, y] >= 0.70
                ):
                    # Shared section wording alone cannot promote a match.
                    score = max(score, 0.25 * overlap + 0.45 * similarity + 0.30 * contextual[x, y])
                scores[x, y] = score
        for x, i in enumerate(left):
            ranked = np.argsort(-scores[x], kind="stable")
            y = int(ranked[0])
            score = scores[x, y]
            if score < 0.35:
                continue
            competing = max((scores[x, k] for k in ranked[1:]), default=0)
            reverse = max((scores[k, y] for k in range(len(left)) if k != x), default=0)
            if (
                (x, y) not in renumbered
                and score >= 0.50
                and score - max(competing, reverse) >= 0.08
            ):
                paired[i] = right[y]
                unmatched.remove(right[y])
            else:
                uncertain[i] = [right[k] for k in ranked if scores[x, k] >= max(0.35, score - 0.08)]

    uncertain = {
        i: available
        for i, options in uncertain.items()
        if (available := [j for j in options if j in unmatched])
    }

    findings = []
    for i, citation in enumerate(before):
        if i in paired:
            replacement = after[paired[i]]
            if citation.text.strip() != replacement.text.strip():
                findings.append(
                    Finding(
                        kind="changed",
                        before=citation,
                        after=replacement,
                        signals=_signals(citation.text, replacement.text),
                    )
                )
        elif i in uncertain:
            best, *alternatives = uncertain[i]
            findings.append(
                Finding(
                    kind="ambiguous",
                    before=citation,
                    after=after[best],
                    alternatives=[after[j] for j in alternatives],
                )
            )
        else:
            findings.append(Finding(kind="removed", before=citation))
    findings.extend(
        Finding(kind="added", after=after[j])
        for j in sorted(unmatched - {j for options in uncertain.values() for j in options})
    )
    return findings
