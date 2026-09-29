from __future__ import annotations

import pytest

from cognivore.rag.embeddings import HashingEmbedder
from cognivore.rag.store import DocumentStore
from cognivore.rag.tokenize import content_tokens, index_terms, tokenize


def test_latin_and_cyrillic_words_are_lowercased_without_punctuation() -> None:
    assert tokenize("Refund policy, 14 days!") == ["refund", "policy", "14", "days"]
    assert tokenize("Политика «возврата».") == ["политика", "возврата"]


def test_cjk_runs_become_overlapping_bigrams() -> None:
    assert tokenize("退款政策") == ["退款", "款政", "政策"]
    assert tokenize("是") == ["是"]


def test_mixed_scripts_split_cleanly() -> None:
    assert tokenize("API限额600次") == ["api", "限额", "600", "次"]


def test_content_tokens_drop_stopwords() -> None:
    assert content_tokens("What is the refund policy?") == ["refund", "policy"]
    assert "какая" not in content_tokens("Какая политика возврата?")


def test_chinese_queries_retrieve_the_matching_chunk() -> None:
    # Regression test: with whitespace tokenization every Chinese query
    # scored exactly 0 against every chunk.
    store = DocumentStore(HashingEmbedder(dim=256), use_approximate_index=False)
    store.add_text("退款政策：首次付款后14天内可全额退款。", source="refund")
    store.add_text("服务等级协议保证99.9%的可用性。", source="sla")
    store.add_text("API限额为每分钟600次请求。", source="api")
    hits = store.search("退款政策是什么？", top_k=1)
    assert hits[0].source == "refund"
    assert hits[0].score > 0


# -- Stemming for retrieval -------------------------------------------------


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("квартале", "квартал"),  # Snowball alone gets this pair wrong
        ("следующем", "следующий"),
        ("тарифы", "тариф"),
        ("компании", "компания"),
        ("шифрования", "шифрование"),
        ("refunds", "refund"),
        ("policies", "policy"),
        ("remboursements", "remboursement"),
        ("Rückerstattungen", "Rückerstattung"),
    ],
)
def test_index_terms_match_other_forms_of_a_word(a: str, b: str) -> None:
    assert index_terms(a) == index_terms(b)


def test_index_terms_keep_different_words_apart() -> None:
    assert index_terms("квартал") != index_terms("квартира")
    assert index_terms("компания") != index_terms("компьютер")


def test_index_terms_leave_numbers_and_cjk_alone() -> None:
    assert index_terms("Старт — 490 рублей") == ["старт", "490", "рубл"]
    assert index_terms("退款政策") == ["退款", "款政", "政策"]
