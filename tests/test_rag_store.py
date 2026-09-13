from __future__ import annotations

from pathlib import Path

from cognivore.rag.embeddings import HashingEmbedder
from cognivore.rag.store import DocumentStore


def test_add_text_and_search_finds_relevant_chunk(document_store: DocumentStore) -> None:
    document_store.add_text(
        "Python is a popular programming language for data science and machine learning.",
        source="doc1.md",
    )
    document_store.add_text(
        "The recipe calls for two cups of flour and a teaspoon of salt.",
        source="doc2.md",
    )
    results = document_store.search("machine learning programming", top_k=1)
    assert len(results) == 1
    assert results[0].source == "doc1.md"


def test_search_on_empty_store_returns_empty_list(document_store: DocumentStore) -> None:
    assert document_store.search("anything", top_k=5) == []


def test_len_reflects_number_of_chunks(document_store: DocumentStore) -> None:
    ids = document_store.add_text("word " * 300, source="doc.md", chunk_size=100, chunk_overlap=10)
    assert len(document_store) == len(ids)
    assert len(ids) > 1


def test_save_and_load_roundtrip(tmp_path: Path, hashing_embedder: HashingEmbedder) -> None:
    store = DocumentStore(embedder=hashing_embedder, use_approximate_index=False)
    store.add_text("Cats are small domesticated carnivorous mammals.", source="animals.md")
    store.save(tmp_path / "store")

    loaded = DocumentStore.load(tmp_path / "store", embedder=hashing_embedder)
    assert len(loaded) == len(store)
    results = loaded.search("small domesticated mammal", top_k=1)
    assert results and results[0].source == "animals.md"


def test_list_chunks_returns_all_chunks(document_store: DocumentStore) -> None:
    document_store.add_text("A short document.", source="a.md")
    document_store.add_text("Another short document.", source="b.md")
    chunks = document_store.list_chunks()
    assert {c["source"] for c in chunks} == {"a.md", "b.md"}


def test_search_multi_finds_hit_from_second_query_when_first_matches_nothing(
    document_store: DocumentStore,
) -> None:
    document_store.add_text(
        "Python is a popular programming language for data science and machine learning.",
        source="doc1.md",
    )
    # The first query shares no words with the ingested text at all (this
    # is standing in for a model translating the query into another
    # language); only the second, closer to the document's own wording,
    # should be able to find it.
    results = document_store.search_multi(
        ["zzz completely unrelated gibberish", "machine learning programming"], top_k=1
    )
    assert len(results) == 1
    assert results[0].source == "doc1.md"


def test_search_multi_deduplicates_and_keeps_best_score(document_store: DocumentStore) -> None:
    document_store.add_text(
        "Python is a popular programming language for data science and machine learning.",
        source="doc1.md",
    )
    results = document_store.search_multi(
        ["machine learning programming", "machine learning programming"], top_k=5
    )
    assert len({r.id for r in results}) == len(results)
