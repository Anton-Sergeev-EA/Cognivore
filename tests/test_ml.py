from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cognivore.ml.clustering import auto_kmeans, kmeans, silhouette_score
from cognivore.ml.gaps import KnowledgeGapTracker
from cognivore.ml.grounding import assess_grounding, split_sentences
from cognivore.ml.insight import build_turn_insight, query_coverage, retrieval_confidence
from cognivore.ml.knowledge_map import KnowledgeMapBuilder
from cognivore.ml.lang import detect_language
from cognivore.ml.projection import fit_pca_2d
from cognivore.ml.topics import cluster_keywords
from cognivore.rag.embeddings import HashingEmbedder
from cognivore.rag.store import DocumentStore


def _blobs(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centres = np.array([[0.0, 0.0, 0.0], [6.0, 6.0, 0.0], [0.0, 6.0, 6.0]])
    x = np.vstack([c + rng.normal(scale=0.4, size=(30, 3)) for c in centres])
    return x, np.repeat(np.arange(3), 30)


# -- projection ----------------------------------------------------------


def test_pca_finds_the_dominant_axis_and_scales_into_unit_box() -> None:
    rng = np.random.default_rng(1)
    x = np.column_stack([rng.normal(scale=10, size=200), rng.normal(scale=1, size=200)])
    pca = fit_pca_2d(x)
    assert abs(abs(pca.components[0, 0]) - 1.0) < 0.05
    assert pca.explained_variance_ratio[0] > 0.9
    coords = pca.transform(x)
    assert np.abs(coords).max() == pytest.approx(1.0)


def test_pca_is_deterministic_in_sign() -> None:
    x, _ = _blobs()
    a = fit_pca_2d(x).transform(x)
    b = fit_pca_2d(x[::-1]).transform(x)
    np.testing.assert_allclose(a, b, atol=1e-9)


def test_pca_handles_a_single_point() -> None:
    pca = fit_pca_2d(np.ones((1, 4)))
    assert pca.transform(np.ones(4)).shape == (1, 2)


# -- clustering ----------------------------------------------------------


def test_kmeans_recovers_well_separated_blobs() -> None:
    x, truth = _blobs()
    result = kmeans(x, 3, seed=0)
    # Same partition up to label permutation.
    for label in range(3):
        assert len(set(result.labels[truth == label])) == 1
    assert len(set(result.labels)) == 3


def test_kmeans_is_deterministic_for_a_seed() -> None:
    x, _ = _blobs()
    a = kmeans(x, 3, seed=7)
    b = kmeans(x, 3, seed=7)
    np.testing.assert_array_equal(a.labels, b.labels)


def test_silhouette_prefers_the_true_partition() -> None:
    x, truth = _blobs()
    shuffled = np.random.default_rng(0).permutation(truth)
    assert silhouette_score(x, truth) > 0.7
    assert silhouette_score(x, shuffled) < 0.1


def test_auto_kmeans_picks_the_number_of_blobs() -> None:
    x, _ = _blobs()
    result, score = auto_kmeans(x, k_min=2, k_max=6, seed=0)
    assert len(set(result.labels)) == 3
    assert score > 0.7


def test_auto_kmeans_on_tiny_input_returns_one_cluster() -> None:
    result, score = auto_kmeans(np.eye(3))
    assert set(result.labels) == {0}
    assert score == 0.0


# -- topics --------------------------------------------------------------


def test_cluster_keywords_name_each_cluster_by_its_distinctive_terms() -> None:
    texts = [
        "refund refund policy money back",
        "refund within fourteen days money",
        "uptime sla availability downtime",
        "sla credits for downtime uptime",
    ]
    keywords = cluster_keywords(texts, [0, 0, 1, 1], top_n=2)
    assert "refund" in keywords[0]
    assert set(keywords[1]) & {"uptime", "downtime", "sla"}


def test_cluster_keywords_skip_overlapping_cjk_bigrams() -> None:
    keywords = cluster_keywords(["退款政策 退款政策 退款"], [0], top_n=3)
    chars = "".join(keywords[0])
    assert len(chars) == len(set(chars))


# -- grounding -----------------------------------------------------------


def test_split_sentences_handles_cjk_punctuation_and_offsets() -> None:
    text = "第一句。第二句！ Third one?"
    spans = split_sentences(text)
    assert [text[s:e] for s, e in spans] == ["第一句。", "第二句！", "Third one?"]


def test_grounding_flags_the_unsupported_sentence() -> None:
    embedder = HashingEmbedder(dim=256)
    passages = ["A full refund is available within 14 days of the first payment."]
    answer = "A full refund is available within 14 days. The moon is made of green cheese."
    report = assess_grounding(answer, passages, embedder)
    assert report is not None
    supported, unsupported = report.sentences
    assert supported.support > 0.6
    assert supported.source_rank == 1
    assert unsupported.support < 0.35
    assert report.level in {"medium", "low"}


def test_grounding_is_not_applicable_without_passages() -> None:
    assert assess_grounding("42", [], HashingEmbedder(dim=16)) is None


# -- language, coverage, gaps -------------------------------------------


@pytest.mark.parametrize(
    ("text", "lang"),
    [
        ("What is the refund policy?", "en"),
        ("Какая политика возврата?", "ru"),
        ("退款政策是什么？", "zh"),
        ("API 限额是多少", "zh"),
        ("12 * 7", "en"),
        ("रिफ़ंड नीति क्या है?", "hi"),
        ("¿Cuál es la política de reembolso?", "es"),
        ("Quelle est la politique de remboursement ?", "fr"),
        ("Wie lautet die Rückerstattungsrichtlinie?", "de"),
        ("Qual è la politica di rimborso?", "it"),
        ("返金ポリシーは？", "ja"),
    ],
)
def test_detect_language(text: str, lang: str) -> None:
    assert detect_language(text) == lang


def test_query_coverage_ignores_cjk_word_boundary_noise() -> None:
    passage = "成长版：每月 999 元，包含 1 TB 存储空间和优先支持。"
    # "长版", "版多", "少钱"... are boundary noise; the meaningful characters
    # (成长版, 包含, 存储...) are what should count.
    assert query_coverage("成长版多少钱？包含什么？", passage) > 0.6
    assert query_coverage("你们支持容器编排吗？", passage) < 0.5


def test_gap_tracker_records_merges_and_resolves(tmp_path: Path) -> None:
    path = tmp_path / "gaps.json"
    tracker = KnowledgeGapTracker(path, threshold=0.3)
    assert tracker.observe("Do you support Kubernetes operators?", 0.1)
    assert tracker.observe("do you support kubernetes operators", 0.2)
    assert not tracker.observe("What is the refund policy?", 0.9)
    items = tracker.items()
    assert len(items) == 1
    assert items[0].count == 2

    reloaded = KnowledgeGapTracker(path, threshold=0.3)
    assert reloaded.items()[0].count == 2
    assert reloaded.resolve(items[0].query)
    assert reloaded.items() == []


# -- end to end over a real store --------------------------------------


@pytest.fixture
def demo_store() -> DocumentStore:
    store = DocumentStore(HashingEmbedder(dim=256), use_approximate_index=False)
    store.add_text("Refund policy: a full refund within 14 days of payment.", source="refund.md")
    store.add_text("Uptime SLA is 99.9 percent; credits for downtime.", source="sla.md")
    store.add_text("API limits: 600 requests per minute on Growth.", source="api.md")
    store.add_text("退款政策：首次付款后14天内可全额退款。", source="refund_zh.md")
    return store


def test_knowledge_map_is_cached_until_the_store_changes(demo_store: DocumentStore) -> None:
    builder = KnowledgeMapBuilder(demo_store)
    first = builder.get()
    assert first.total_chunks == 4
    assert len(first.points) == 4
    assert all(-1.0001 <= p.x <= 1.0001 and -1.0001 <= p.y <= 1.0001 for p in first.points)
    assert builder.get() is first
    demo_store.add_text("Webhooks fire on upload completed.", source="hooks.md")
    assert builder.get() is not first
    assert builder.get().total_chunks == 5


def test_knowledge_map_samples_large_stores() -> None:
    store = DocumentStore(HashingEmbedder(dim=32), use_approximate_index=False)
    for i in range(40):
        store.add_text(f"note number {i} about topic {i % 4}", source=f"{i}.md")
    kmap = KnowledgeMapBuilder(store, max_points=15).get()
    assert kmap.total_chunks == 40
    assert len(kmap.points) == 15


def test_turn_insight_for_a_supported_answer(demo_store: DocumentStore, tmp_path: Path) -> None:
    gaps = KnowledgeGapTracker(tmp_path / "gaps.json")
    insight = build_turn_insight(
        demo_store,
        question="What is the refund policy?",
        answer="A full refund is possible within 14 days of payment.",
        tool_queries=["refund policy"],
        used_knowledge_base=True,
        map_builder=KnowledgeMapBuilder(demo_store),
        gap_tracker=gaps,
    )
    assert insight.language == "en"
    assert insight.hits[0].source == "refund.md"
    assert insight.hits[0].rank == 1
    assert insight.query_point is not None
    assert insight.grounding is not None and insight.grounding.level == "high"
    assert not insight.gap
    assert gaps.items() == []


def test_turn_insight_records_a_gap(demo_store: DocumentStore, tmp_path: Path) -> None:
    gaps = KnowledgeGapTracker(tmp_path / "gaps.json")
    insight = build_turn_insight(
        demo_store,
        question="Do you support Kubernetes operators?",
        answer="I don't know.",
        tool_queries=[],
        used_knowledge_base=True,
        gap_tracker=gaps,
    )
    assert insight.gap
    assert retrieval_confidence("Do you support Kubernetes operators?", []) == 0.0
    assert gaps.items()[0].query == "Do you support Kubernetes operators?"


def test_turn_insight_skips_grounding_when_no_search_was_used(demo_store: DocumentStore) -> None:
    insight = build_turn_insight(
        demo_store,
        question="What is 6 * 7?",
        answer="42",
        tool_queries=[],
        used_knowledge_base=False,
    )
    assert insight.grounding is None
    assert not insight.gap


def test_a_sentence_naming_the_source_is_a_citation_not_an_unsupported_claim() -> None:
    embedder = HashingEmbedder(dim=256)
    passages = [
        "Refund policy: a full refund is available within 14 days of the first payment.",
        "SLA: guaranteed uptime is 99.9 percent.",
    ]
    sources = ["Skylark Cloud demo knowledge base (EN)", "NordCloud demo knowledge base (RU)"]
    for answer, cite in [
        (
            "A full refund is available within 14 days. "
            "This information is from (Skylark Cloud demo knowledge base (EN)).",
            "en",
        ),
        (
            "Полный возврат возможен в течение 14 дней. "
            "Это информация из (Skylark Cloud демонстрационной базы знаний (EN)).",
            "ru",
        ),
    ]:
        report = assess_grounding(answer, passages, embedder, sources=sources)
        assert report is not None
        last = report.sentences[-1]
        assert last.citation, cite
        assert last.source_rank == 1
        assert last.support >= 0.35  # never underlined


def test_a_real_claim_that_mentions_the_source_is_still_checked() -> None:
    embedder = HashingEmbedder(dim=256)
    report = assess_grounding(
        "Skylark Cloud gives every customer free lifetime storage and a personal robot butler.",
        ["Refund policy: a full refund is available within 14 days of the first payment."],
        embedder,
        sources=["Skylark Cloud demo knowledge base (EN)"],
    )
    assert report is not None
    assert not report.sentences[0].citation
    assert report.sentences[0].support < 0.35


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Do you have a mobile app for iPhone?", True),
        ("Есть ли у вас мобильное приложение для iPhone?", True),
        ("你们有苹果手机应用吗？", True),
        ("Gibt es eine Handy-App für das iPhone?", True),
        ("What is 15% of 149?", False),
        ("Сколько будет 15% от 4900?", False),
        ("12 * 7", False),
        ("How are you?", False),
        ("Привет", False),
    ],
)
def test_looks_like_knowledge_question(text: str, expected: bool) -> None:
    from cognivore.ml.intent import looks_like_knowledge_question

    assert looks_like_knowledge_question(text) is expected


def test_a_gap_carries_no_misleading_sources_or_grounding(
    demo_store: DocumentStore, tmp_path: Path
) -> None:
    insight = build_turn_insight(
        demo_store,
        question="Do you support Kubernetes operators?",
        answer="The knowledge base has no information on Kubernetes operators.",
        tool_queries=[],
        used_knowledge_base=True,
        map_builder=KnowledgeMapBuilder(demo_store),
        gap_tracker=KnowledgeGapTracker(tmp_path / "gaps.json"),
    )
    assert insight.gap
    assert insight.hits == []
    assert insight.grounding is None
    assert insight.query_point is not None  # the question is still placed on the map


def test_query_coverage_accepts_other_forms_of_a_word() -> None:
    """A Russian question rarely uses the passage's exact word forms:
    "тариф" / "Тарифы", "шифрование" / "Шифрование". Before, such a
    question about an uploaded video's slide counted as unanswered."""
    passage = "На экране: Тарифы · Старт — 490 рублей в месяц · Бизнес — 1990 рублей в месяц"
    assert query_coverage("Сколько стоит тариф Старт?", passage) >= 0.5
    assert query_coverage("What do the refunds cover?", "Refund policy: a full refund") > 0.3
    # ...but a shared beginning alone doesn't make two words the same one.
    assert query_coverage("компьютер", "компания") == 0.0
