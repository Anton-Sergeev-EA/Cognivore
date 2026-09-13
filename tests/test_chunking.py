from __future__ import annotations

import pytest

from cognivore.rag.chunking import split_text


def test_split_text_short_text_single_chunk() -> None:
    chunks = split_text("hello world", chunk_size=100, chunk_overlap=10)
    assert len(chunks) == 1
    assert chunks[0].text == "hello world"


def test_split_text_respects_chunk_size_upper_bound() -> None:
    text = " ".join(f"word{i}" for i in range(500))
    chunks = split_text(text, chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.text) <= 100


def test_split_text_prefers_paragraph_boundaries() -> None:
    text = "Paragraph one is here.\n\nParagraph two is here.\n\nParagraph three."
    chunks = split_text(text, chunk_size=40, chunk_overlap=5)
    joined = " ".join(c.text for c in chunks)
    assert "Paragraph one" in joined
    assert "Paragraph three" in joined


def test_split_text_rejects_overlap_ge_chunk_size() -> None:
    with pytest.raises(ValueError):
        split_text("some text", chunk_size=50, chunk_overlap=50)


def test_split_text_empty_string_returns_no_chunks() -> None:
    assert split_text("", chunk_size=100, chunk_overlap=10) == []
