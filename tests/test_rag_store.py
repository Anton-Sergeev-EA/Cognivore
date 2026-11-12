from __future__ import annotations

import json
from pathlib import Path

import pytest

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


def test_save_and_load_keeps_vectors_without_reembedding(
    tmp_path: Path, hashing_embedder: HashingEmbedder
) -> None:
    store = DocumentStore(embedder=hashing_embedder, use_approximate_index=False)
    store.add_text("Cats are small domesticated carnivorous mammals.", source="animals.md")
    store.save(tmp_path / "store")
    assert (tmp_path / "store" / "vectors.npy").exists()

    loaded = DocumentStore.load(tmp_path / "store", embedder=hashing_embedder)
    assert not loaded.reindexed
    _ids, matrix = loaded.vector_matrix()
    _, original = store.vector_matrix()
    assert matrix.shape == original.shape
    assert (matrix == original).all()


def test_load_with_a_different_embedder_reembeds(tmp_path: Path) -> None:
    # Regression guard: vectors from one embedder compared against queries
    # from another silently return garbage, so a mismatch must re-embed.
    store = DocumentStore(embedder=HashingEmbedder(dim=64), use_approximate_index=False)
    store.add_text("Cats are small domesticated carnivorous mammals.", source="animals.md")
    store.save(tmp_path / "store")

    other = HashingEmbedder(dim=64, ngram_range=(1, 1))
    loaded = DocumentStore.load(tmp_path / "store", embedder=other)
    assert loaded.reindexed
    hits = loaded.search("domesticated mammals", top_k=1)
    assert hits and hits[0].source == "animals.md"


def test_load_of_a_store_saved_before_vectors_were_persisted(
    tmp_path: Path, hashing_embedder: HashingEmbedder
) -> None:
    store = DocumentStore(embedder=hashing_embedder, use_approximate_index=False)
    store.add_text("Cats are small domesticated carnivorous mammals.", source="animals.md")
    store.save(tmp_path / "store")
    (tmp_path / "store" / "vectors.npy").unlink()

    loaded = DocumentStore.load(tmp_path / "store", embedder=hashing_embedder)
    assert loaded.reindexed
    assert loaded.search("mammals", top_k=1)[0].source == "animals.md"


def test_version_changes_on_every_add(document_store: DocumentStore) -> None:
    before = document_store.version
    document_store.add_text("One more document.", source="x.md")
    assert document_store.version == before + 1


def test_a_question_in_other_word_forms_finds_its_passage() -> None:
    """Live finding: "Куда компания выходит в следующем квартале?" ranked a
    passage that merely said "компания" above the one about "Планы на
    следующий квартал" -- lexical search saw no shared word in the latter."""
    store = DocumentStore(HashingEmbedder())
    store.add_text("О компании: компания обслуживает 340 клиентов.", source="about")
    store.add_text("Планы на следующий квартал: запуск в Казахстане.", source="plans")

    hits = store.search("Куда компания выходит в следующем квартале?", top_k=2)

    assert hits[0].source == "plans"


def test_nsw_store_migrates_using_saved_vectors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cognivore.index import HNSWIndex, NSWIndex
    from cognivore.rag import store as store_module

    if HNSWIndex is None or NSWIndex is None:
        pytest.skip("native indexes are required")
    embedder = HashingEmbedder(dim=32)
    with monkeypatch.context() as legacy:
        legacy.setattr(store_module, "HNSWIndex", None)
        old_store = DocumentStore(embedder)
        old_store.add_text("The compressor has three stages.", source="manual.md")
        old_store.save(tmp_path)
    meta_file = tmp_path / "meta.json"
    meta = json.loads(meta_file.read_text())
    assert meta.pop("index_backend") == "nsw"
    meta["format"] = 2
    meta_file.write_text(json.dumps(meta))

    with monkeypatch.context() as migration:
        migration.setattr(embedder, "embed", lambda _texts: pytest.fail("vectors were re-embedded"))
        upgraded = DocumentStore.load(tmp_path, embedder)
    assert upgraded.index_rebuilt and not upgraded.reindexed
    assert upgraded.search("compressor")[0].source == "manual.md"
    upgraded.save(tmp_path)
    assert (tmp_path / "index.bin").read_bytes().startswith(b"HNSW")

    restored = DocumentStore.load(tmp_path, embedder)
    assert not restored.index_rebuilt and not restored.reindexed
    assert restored.search("compressor") == upgraded.search("compressor")

    # A reader without the extension can search the same saved store.
    with monkeypatch.context() as fallback:
        fallback.setattr(store_module, "HNSWIndex", None)
        fallback.setattr(store_module, "NSWIndex", None)
        fallback.setattr(store_module, "FlatIndex", store_module.FlatIndexPy)
        exact = DocumentStore.load(tmp_path, embedder)
        assert not exact.index_rebuilt and not exact.reindexed
        assert exact.search("compressor")[0].source == "manual.md"

    (tmp_path / "index.bin").write_bytes(b"HNSW")
    repaired = DocumentStore.load(tmp_path, embedder)
    assert repaired.index_rebuilt and not repaired.reindexed
    assert repaired.search("compressor")[0].source == "manual.md"
