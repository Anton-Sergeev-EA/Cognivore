from __future__ import annotations

from cognivore.rag.snippets import best_snippet, split_sentences

_LONG = (
    "About the company. Skylark Cloud hosts video for online courses. "
    "Pricing plans are listed below and change every year. "
    "Uptime is guaranteed at 99.9 percent on Growth. "
    "Refund policy: a full refund is available within 14 days of the first payment. "
    "After 14 days the subscription can be paused for up to 3 months. "
    "Webhooks fire when an upload completes."
)


def test_short_text_is_returned_whole() -> None:
    assert best_snippet("Short text.", "anything", limit=300) == "Short text."


def test_snippet_is_centred_on_the_matching_sentence() -> None:
    snippet = best_snippet(_LONG, "What is the refund policy?", limit=120)
    assert "full refund is available within 14 days" in snippet
    assert len(snippet) <= 120
    assert not snippet.startswith("About the company")


def test_snippet_falls_back_to_the_beginning_without_a_match() -> None:
    assert best_snippet(_LONG, "kubernetes", limit=50) == _LONG[:50]


def test_chinese_snippet_finds_the_refund_section() -> None:
    text = (
        "安全与合规：所有视频在存储时使用 AES-256 加密。每 6 小时进行一次备份。"
        "集成：支持钉钉、企业微信和飞书。"
        "退款政策：首次付款后 14 天内可以申请全额退款。超过 14 天后不再退款。"
    )
    snippet = best_snippet(text, "退款政策是什么？", limit=40)
    assert "全额退款" in snippet


def test_split_sentences_trims_whitespace() -> None:
    text = "  One.  Two!\nThree"
    assert [text[s:e] for s, e in split_sentences(text)] == ["One.", "Two!", "Three"]


def test_lines_without_terminal_punctuation_are_sentences_too() -> None:
    # Regression test: headings and wrapped lines without a final period
    # used to be silently dropped.
    text = "## Refund policy\nA full refund within\n14 days. Done"
    assert [text[s:e] for s, e in split_sentences(text)] == [
        "## Refund policy",
        "A full refund within",
        "14 days.",
        "Done",
    ]


def test_snippet_does_not_open_with_the_preceding_section() -> None:
    text = (
        "## API limits\nGrowth: 600 requests per minute. Enterprise: negotiated limits "
        "starting at 3000 requests per minute and a dedicated account manager.\n"
        "## Refund policy\nA full refund is available within 14 days of the first payment."
    )
    snippet = best_snippet(text, "What is the refund policy?", limit=120)
    assert snippet.startswith("## Refund policy")
