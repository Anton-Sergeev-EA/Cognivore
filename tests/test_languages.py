"""End-to-end check that every UI language actually works in the offline
demo: each example question shown in the UI (i18n.js) is answered from the
demo handbook in *that* language without being flagged as a knowledge gap,
and a question none of the handbooks can answer lands on the gap radar."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from cognivore.config import Settings
from cognivore.web.app import create_app

_I18N = (Path(__file__).parent.parent / "src/cognivore/web/static/i18n.js").read_text("utf-8")
LANGS = json.loads(re.search(r"SUPPORTED_LANGS = (\[.*?\]);", _I18N).group(1))  # type: ignore[union-attr]


def _examples(lang: str) -> list[str]:
    block = _I18N[_I18N.index(f"\n  {lang}: {{") :]
    body = re.search(r"examples:\s*\[(.*?)\]", block, re.S).group(1)  # type: ignore[union-attr]
    return json.loads("[" + body.strip().rstrip(",") + "]")


# A question none of the demo handbooks answers, in every UI language.
_UNANSWERABLE = {
    "ru": "Есть ли у вас мобильное приложение для iPhone?",
    "en": "Do you have a mobile app for iPhone?",
    "zh": "你们有苹果手机应用吗？",
    "es": "¿Tienen una aplicación móvil para iPhone?",
    "hi": "क्या आपके पास iPhone के लिए मोबाइल ऐप है?",
    "fr": "Avez-vous une application mobile pour iPhone ?",
    "de": "Gibt es eine Handy-App für das iPhone?",
    "ja": "iPhone 向けのスマホアプリはありますか？",
    "it": "Avete un'app per cellulare per iPhone?",
}


@pytest.fixture(scope="module")
def client(tmp_path_factory: pytest.TempPathFactory) -> TestClient:
    settings = Settings(
        data_dir=tmp_path_factory.mktemp("langs"),
        prefer_semantic_embedder=False,
        seed_demo_kb=True,
    )
    return TestClient(create_app(settings))


def test_ui_ships_nine_languages_with_russian_first() -> None:
    assert LANGS == ["ru", "en", "zh", "es", "hi", "fr", "de", "ja", "it"]
    assert set(_UNANSWERABLE) == set(LANGS)


def test_every_language_has_a_demo_handbook(client: TestClient) -> None:
    sources = {c["source"] for c in client.get("/api/knowledge_base").json()}
    for lang in LANGS:
        assert any(s.endswith(f"({lang.upper()})") for s in sources), lang


@pytest.mark.parametrize("lang", LANGS)
def test_example_questions_are_answered_in_their_own_language(
    client: TestClient, lang: str
) -> None:
    for question in _examples(lang):
        body = client.post("/api/chat", json={"message": question}).json()
        insight = body["insight"]
        if not insight["used_knowledge_base"]:  # the percentage example
            assert body["answer"].strip().replace(".", "").isdigit(), question
            continue
        assert insight["hits"][0]["source"].endswith(f"({lang.upper()})"), question
        assert not insight["gap"], question
        assert insight["grounding"]["level"] == "high", question


@pytest.mark.parametrize("lang", LANGS)
def test_unanswerable_question_is_a_gap_in_every_language(client: TestClient, lang: str) -> None:
    from cognivore.agent.replies import NOT_IN_KNOWLEDGE_BASE

    body = client.post("/api/chat", json={"message": _UNANSWERABLE[lang]}).json()
    insight = body["insight"]
    assert insight["used_knowledge_base"], lang
    assert insight["gap"], (lang, insight["confidence"])
    # An honest "not in the knowledge base", in the question's language --
    # not the least-unrelated passage presented as an answer.
    assert body["answer"].strip() == NOT_IN_KNOWLEDGE_BASE[lang], lang
    assert insight["hits"] == [] and insight["grounding"] is None
