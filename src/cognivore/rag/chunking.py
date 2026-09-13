"""Text chunking for RAG ingestion.

Implements a recursive character splitter (the same idea popularized by
LangChain's ``RecursiveCharacterTextSplitter``): try to split on the
"biggest" separator first (paragraph breaks), and only fall back to finer
separators (sentences, words, characters) for pieces that are still too
large. This keeps chunks aligned to natural text boundaries far more often
than a naive fixed-width slice, which materially improves retrieval
quality since a chunk that ends mid-sentence pollutes its own embedding.
"""

from __future__ import annotations

from dataclasses import dataclass

_DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]


@dataclass(frozen=True)
class Chunk:
    text: str
    start: int  # character offset in the original document
    end: int


def split_text(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    separators: list[str] | None = None,
) -> list[Chunk]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    seps = separators if separators is not None else _DEFAULT_SEPARATORS
    raw_pieces = _recursive_split(text, chunk_size, seps)
    return _merge_with_overlap(text, raw_pieces, chunk_size, chunk_overlap)


def _recursive_split(text: str, chunk_size: int, separators: list[str]) -> list[str]:
    if len(text) <= chunk_size or not text:
        return [text] if text else []

    sep, remaining_seps = separators[0], separators[1:]
    if sep == "":
        # Last resort: hard character slice.
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    pieces = text.split(sep)
    out: list[str] = []
    for piece in pieces:
        if not piece:
            continue
        if len(piece) > chunk_size and remaining_seps:
            out.extend(_recursive_split(piece, chunk_size, remaining_seps))
        else:
            out.append(piece)
    return out


def _merge_with_overlap(
    original: str, pieces: list[str], chunk_size: int, chunk_overlap: int
) -> list[Chunk]:
    """Greedily packs the small, separator-aligned ``pieces`` back together
    up to ``chunk_size``, sliding the window back by roughly
    ``chunk_overlap`` characters' worth of *whole pieces* between chunks so
    a sentence/paragraph boundary is never split mid-way just to satisfy the
    overlap.

    This walks ``pieces`` by index (rather than mutating/truncating
    already-assembled strings) specifically so that content is never
    silently dropped -- an earlier, string-splicing version of this
    function could truncate the tail of a chunk to make room for the
    carried-over overlap, which lost real text. Operating on whole pieces
    and only ever *extending* a chunk (never slicing one) avoids that class
    of bug entirely.
    """
    if not pieces:
        return []

    texts: list[str] = []
    n = len(pieces)
    start_idx = 0
    while start_idx < n:
        current = ""
        end_idx = start_idx
        while end_idx < n:
            candidate = f"{current} {pieces[end_idx]}".strip() if current else pieces[end_idx]
            if len(candidate) > chunk_size and current:
                break
            current = candidate
            end_idx += 1
            if len(current) >= chunk_size:
                break
        texts.append(current)

        if end_idx >= n:
            break

        # Slide back from end_idx by whole pieces until we've covered
        # ~chunk_overlap characters, so the next chunk repeats that tail.
        back_idx = end_idx
        overlap_len = 0
        while back_idx > start_idx and overlap_len < chunk_overlap:
            back_idx -= 1
            overlap_len += len(pieces[back_idx]) + 1
        start_idx = back_idx if back_idx > start_idx else end_idx  # guarantee forward progress

    result: list[Chunk] = []
    search_from = 0
    for text in texts:
        anchor = text[:30]
        idx = original.find(anchor, search_from) if anchor else search_from
        idx = max(idx, 0)
        result.append(Chunk(text=text, start=idx, end=min(idx + len(text), len(original))))
        # Allow the next chunk's anchor to be found even inside the overlap
        # region of this one, rather than skipping past it.
        search_from = max(0, idx - chunk_overlap)
    return result
