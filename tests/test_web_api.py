from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from cognivore.config import Settings
from cognivore.web.app import create_app


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    settings = Settings(data_dir=tmp_path / ".cognivore", prefer_semantic_embedder=False)
    app = create_app(settings)
    return TestClient(app)


def test_seed_demo_kb_seeds_a_fresh_store_on_startup(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path / ".cognivore",
        prefer_semantic_embedder=False,
        seed_demo_kb=True,
    )
    app = create_app(settings)
    client = TestClient(app)

    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json()["knowledge_base_chunks"] > 0


def test_seed_demo_kb_never_touches_an_already_populated_store(tmp_path: Path) -> None:
    data_dir = tmp_path / ".cognivore"

    # First server run: seeding is off, but the user ingests their own doc.
    settings = Settings(data_dir=data_dir, prefer_semantic_embedder=False, seed_demo_kb=False)
    app = create_app(settings)
    app.state.store.add_text("my own private notes", source="notes.md")
    app.state.store.save(data_dir / "store")
    own_chunk_count = len(app.state.store)

    # Second run: seeding is now on, but the store already has content, so
    # it must be left alone rather than having demo docs merged in.
    settings2 = Settings(data_dir=data_dir, prefer_semantic_embedder=False, seed_demo_kb=True)
    app2 = create_app(settings2)
    client2 = TestClient(app2)
    res = client2.get("/api/health")
    assert res.json()["knowledge_base_chunks"] == own_chunk_count


def test_health_endpoint(client: TestClient) -> None:
    res = client.get("/api/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["llm_backend"] == "FakeLLMBackend"
    assert "calculator" in body["tools"]


def test_chat_endpoint_arithmetic(client: TestClient) -> None:
    res = client.post("/api/chat", json={"message": "What is 3 * 3?"})
    assert res.status_code == 200
    body = res.json()
    assert body["answer"].strip() == "9"


def test_chat_endpoint_rejects_empty_message(client: TestClient) -> None:
    res = client.post("/api/chat", json={"message": "   "})
    assert res.status_code == 400


def test_ingest_file_then_chat_finds_it(client: TestClient) -> None:
    files = {"file": ("notes.md", b"The launch code is banana42.", "text/markdown")}
    res = client.post("/api/ingest/file", files=files)
    assert res.status_code == 200
    assert res.json()["chunks_added"] == 1

    kb = client.get("/api/knowledge_base").json()
    assert any("banana42" in c["text"] for c in kb)


def test_chat_stream_delivers_the_final_answer(client: TestClient) -> None:
    with client.stream("GET", "/api/chat/stream", params={"message": "What is 3 * 3?"}) as res:
        body = res.read().decode()
    assert "answer_chunk" in body
    assert "9" in body


def test_chat_stream_delivers_answer_even_without_a_final_answer_marker(client: TestClient) -> None:
    # Regression test for a real bug found live: a real model doesn't
    # always use the literal "Final Answer:" marker (sometimes it just
    # answers directly). run_stream still recognizes that as the final
    # answer, but nothing was ever streamed live for it -- without a
    # fallback in the SSE bridge, that answer silently never reached the
    # client at all.
    class _DirectAnswerLLM:
        def generate(self, messages, config):
            return "Hello there"

        def stream(self, messages, config):
            yield "Hello there"

    client.app.state.agent.llm = _DirectAnswerLLM()

    with client.stream("GET", "/api/chat/stream", params={"message": "hi"}) as res:
        body = res.read().decode()
    assert "Hello there" in body


def test_ingest_file_rejects_non_utf8(client: TestClient) -> None:
    files = {"file": ("bad.bin", b"\xff\xfe\x00\x01", "application/octet-stream")}
    res = client.post("/api/ingest/file", files=files)
    assert res.status_code == 400


@pytest.mark.skipif(
    importlib.util.find_spec("faster_whisper") is not None,
    reason="only meaningful when the `audio` extra is NOT installed",
)
def test_media_audio_endpoint_returns_503_when_extra_not_installed(client: TestClient) -> None:
    # When the `audio` extra isn't installed, the tool is never registered
    # and the endpoint should say so explicitly rather than attempt (and
    # fail) to transcribe.
    res = client.post("/api/media/audio", files={"file": ("a.wav", b"\x00\x00", "audio/wav")})
    assert res.status_code == 503


def test_index_page_served(client: TestClient) -> None:
    res = client.get("/")
    assert res.status_code == 200
    assert b"Cognivore" in res.content


def test_static_assets_are_not_cached(client: TestClient) -> None:
    # Regression test for a real bug found live: plain StaticFiles sends no
    # Cache-Control header, so a browser can go on serving an old cached
    # copy of app.js/i18n.js/etc. after the file on disk (and its
    # Last-Modified) has changed -- a UI update landed on the server but
    # stayed invisible in an already-open browser, indistinguishable from
    # the deploy having silently failed. Every static response must tell
    # the browser not to do that.
    for path in ("/", "/app.js", "/i18n.js", "/styles.css", "/cortex.js"):
        res = client.get(path)
        assert res.headers.get("cache-control") == "no-store", path


# -- explainability endpoints (cognivore.ml) -------------------------------


def _sse_events(body: str) -> dict[str, list[str]]:
    """Parses an SSE stream the way a browser's EventSource does: several
    ``data:`` lines of one event are joined with newlines."""
    events: dict[str, list[str]] = {}
    name: str | None = None
    data: list[str] = []
    for line in [*body.splitlines(), ""]:
        if not line:
            if name is not None:
                events.setdefault(name, []).append("\n".join(data))
            name, data = None, []
        elif line.startswith("event:"):
            name = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            value = line[5:]
            data.append(value[1:] if value.startswith(" ") else value)
    return events


@pytest.fixture
def seeded_client(tmp_path: Path) -> TestClient:
    settings = Settings(
        data_dir=tmp_path / ".cognivore", prefer_semantic_embedder=False, seed_demo_kb=True
    )
    return TestClient(create_app(settings))


def test_health_reports_demo_mode_and_embedder(client: TestClient) -> None:
    body = client.get("/api/health").json()
    assert body["demo_mode"] is True
    assert body["embedder"].startswith("hashing")


def test_seeded_store_includes_all_three_demo_languages(seeded_client: TestClient) -> None:
    sources = {c["source"] for c in seeded_client.get("/api/knowledge_base").json()}
    assert any("(EN)" in s for s in sources)
    assert any("(RU)" in s for s in sources)
    assert any("(ZH)" in s for s in sources)


def test_knowledge_map_endpoint(seeded_client: TestClient) -> None:
    body = seeded_client.get("/api/insights/map").json()
    assert body["total_chunks"] == len(body["points"]) > 0
    assert body["clusters"]
    cluster_ids = {c["id"] for c in body["clusters"]}
    assert all(p["cluster"] in cluster_ids for p in body["points"])
    assert sum(c["size"] for c in body["clusters"]) == len(body["points"])


def test_knowledge_map_of_an_empty_store(client: TestClient) -> None:
    body = client.get("/api/insights/map").json()
    assert body["points"] == [] and body["total_chunks"] == 0


@pytest.mark.parametrize(
    "question",
    ["Какая политика возврата средств?", "退款政策是什么？", "What is the refund policy?"],
)
def test_stream_emits_an_insight_for_knowledge_questions(
    seeded_client: TestClient, question: str
) -> None:
    with seeded_client.stream("GET", "/api/chat/stream", params={"message": question}) as res:
        events = _sse_events(res.read().decode())
    assert "answer_final" in events
    insight = json.loads(events["insight"][0])
    assert insight["used_knowledge_base"] is True
    assert insight["hits"] and insight["hits"][0]["rank"] == 1
    assert insight["query_point"] is not None
    assert insight["grounding"] is not None
    assert insight["gap"] is False
    # Offsets must index into the exact final answer text.
    answer = events["answer_final"][0]
    for sentence in insight["grounding"]["sentences"]:
        assert 0 <= sentence["start"] < sentence["end"] <= len(answer)


def test_unanswerable_question_lands_on_the_gap_radar(seeded_client: TestClient) -> None:
    question = "Do you support Kubernetes operators?"
    with seeded_client.stream("GET", "/api/chat/stream", params={"message": question}) as res:
        events = _sse_events(res.read().decode())
    assert json.loads(events["insight"][0])["gap"] is True

    gaps = seeded_client.get("/api/insights/gaps").json()
    assert [g["query"] for g in gaps] == [question]
    assert seeded_client.delete("/api/insights/gaps", params={"query": question}).status_code == 200
    assert seeded_client.get("/api/insights/gaps").json() == []
    assert seeded_client.delete("/api/insights/gaps", params={"query": question}).status_code == 404


def test_chat_endpoint_includes_insight(seeded_client: TestClient) -> None:
    body = seeded_client.post("/api/chat", json={"message": "What is the refund policy?"}).json()
    assert body["insight"]["hits"]
    arithmetic = seeded_client.post("/api/chat", json={"message": "What is 6 * 7?"}).json()
    assert arithmetic["answer"].strip() == "42"
    assert arithmetic["insight"]["used_knowledge_base"] is False
    assert arithmetic["insight"]["grounding"] is None


def test_ingest_text_takes_a_json_body(client: TestClient) -> None:
    res = client.post(
        "/api/ingest/text", json={"source": "k8s.md", "text": "We ship a Helm chart."}
    )
    assert res.status_code == 200
    assert res.json()["chunks_added"] == 1
    assert client.post("/api/ingest/text", json={"source": "x", "text": "  "}).status_code == 400


def test_single_chunk_endpoint(client: TestClient) -> None:
    client.post("/api/ingest/text", json={"source": "a.md", "text": "Alpha beta gamma."})
    chunk_id = client.get("/api/knowledge_base").json()[0]["id"]
    body = client.get(f"/api/knowledge_base/{chunk_id}").json()
    assert body["source"] == "a.md"
    assert client.get("/api/knowledge_base/999999").status_code == 404


def test_oversized_upload_is_rejected(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path / ".cognivore", prefer_semantic_embedder=False, max_document_upload_mb=1
    )
    client = TestClient(create_app(settings))
    big = b"a" * (1024 * 1024 + 1)
    res = client.post("/api/ingest/file", files={"file": ("big.md", big, "text/markdown")})
    assert res.status_code == 413


def test_new_static_assets_are_served(client: TestClient) -> None:
    res = client.get("/cortex.js")
    assert res.status_code == 200
    assert res.headers.get("cache-control") == "no-store"


def test_auto_context_explains_an_answer_given_without_calling_search(
    seeded_client: TestClient,
) -> None:
    # The live failure this guards against: qwen2.5:3b answered a knowledge
    # question straight away, never calling search_knowledge_base, and
    # invented details -- so the UI had no sources and no grounding to show.
    from cognivore.bootstrap import build_context_provider

    class _NoSearchLLM:
        def generate(self, messages, config):
            return (
                "Final Answer: A full refund is available within 14 days of the first "
                "payment. You must fill in a special form on our website."
            )

        def stream(self, messages, config):
            yield self.generate(messages, config)

    app = seeded_client.app
    app.state.agent.llm = _NoSearchLLM()
    app.state.agent.context_provider = build_context_provider(app.state.store, app.state.settings)
    body = seeded_client.post("/api/chat", json={"message": "What is the refund policy?"}).json()

    assert body["trace"][0]["action"] == "search_knowledge_base"
    insight = body["insight"]
    assert insight["used_knowledge_base"]
    assert insight["hits"][0]["source"].endswith("(EN)")
    supported, invented = insight["grounding"]["sentences"]
    assert supported["support"] >= 0.35
    assert invented["support"] < 0.35  # the made-up "special form" gets underlined


def test_documents_ingested_into_a_fresh_empty_store_are_searchable(client: TestClient) -> None:
    # Regression test: an empty store is falsy, and build_agent's old
    # `store or build_document_store(...)` gave the agent a separate store,
    # so documents uploaded to a fresh install were never found in chat.
    assert client.app.state.agent.tools.get("search_knowledge_base").store is (
        client.app.state.store
    )
    files = {"file": ("notes.md", b"The launch code is banana42.", "text/markdown")}
    client.post("/api/ingest/file", files=files)
    body = client.post(
        "/api/chat", json={"message": "search the knowledge base for the launch code"}
    ).json()
    assert "banana42" in body["answer"]


def test_real_model_gap_question_is_recorded_and_shows_no_fake_sources(
    seeded_client: TestClient,
) -> None:
    # Live finding with qwen2.5:3b: for a question the knowledge base can't
    # answer, the model either told the user to "search the internet",
    # relayed an "unknown tool" error, or answered with its own
    # meta-reasoning. In strict mode (the default) Cognivore answers such a
    # question itself, without calling the model.
    from cognivore.bootstrap import build_context_provider

    class _MustNotBeCalled:
        calls = 0

        def generate(self, messages, config):
            _MustNotBeCalled.calls += 1
            return "Final Answer: made up"

        def stream(self, messages, config):
            yield self.generate(messages, config)

    app = seeded_client.app
    app.state.agent.llm = _MustNotBeCalled()
    app.state.agent.context_provider = build_context_provider(app.state.store, app.state.settings)
    assert app.state.agent.strict_knowledge_answers
    question = "Есть ли у вас мобильное приложение для iPhone?"
    with seeded_client.stream("GET", "/api/chat/stream", params={"message": question}) as res:
        events = _sse_events(res.read().decode())

    assert _MustNotBeCalled.calls == 0
    assert events["answer_final"] == ["В базе знаний нет информации об этом."]
    assert json.loads(events["trace"][0])["observation"] == "No relevant passages found."
    insight = json.loads(events["insight"][0])
    assert insight["gap"] is True
    assert insight["hits"] == [] and insight["grounding"] is None
    assert [g["query"] for g in seeded_client.get("/api/insights/gaps").json()] == [question]


def test_non_strict_mode_tells_the_model_the_knowledge_base_has_nothing(
    seeded_client: TestClient,
) -> None:
    from cognivore.bootstrap import build_context_provider

    class _HonestLLM:
        system = ""

        def generate(self, messages, config):
            _HonestLLM.system = "\n".join(m.content for m in messages if m.role == "system")
            return "Final Answer: The knowledge base has no information about a mobile app."

        def stream(self, messages, config):
            yield self.generate(messages, config)

    agent = seeded_client.app.state.agent
    agent.llm = _HonestLLM()
    agent.strict_knowledge_answers = False
    agent.context_provider = build_context_provider(
        seeded_client.app.state.store, seeded_client.app.state.settings
    )
    body = seeded_client.post(
        "/api/chat", json={"message": "Do you have a mobile app for iPhone?"}
    ).json()
    assert "contains nothing about it" in _HonestLLM.system
    assert body["answer"].startswith("The knowledge base has no information")
    assert body["insight"]["gap"] is True


# -- Audio and video ------------------------------------------------------


def _fake_video_analysis(path: str) -> object:
    from cognivore.media.audio import Transcript, TranscriptSegment
    from cognivore.media.video import Keyframe, VideoAnalysis

    return VideoAnalysis(
        duration=20.0,
        scenes=[
            Keyframe(0.0, 0, "Тарифы\nСтарт — 490 рублей в месяц", end=10.0),
            Keyframe(10.0, 250, "Безопасность\nШифрование AES-256", end=20.0),
        ],
        transcript=Transcript(
            text="",
            segments=[TranscriptSegment(12.0, 16.0, "Все данные шифруются алгоритмом AES.")],
            language="ru",
        ),
        ocr_languages="rus+eng",
    )


@pytest.fixture
def video_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    pytest.importorskip("cv2", reason="the video tool needs the `video` extra")
    monkeypatch.setattr(
        "cognivore.web.app.analyze_video", lambda path, **_: _fake_video_analysis(path)
    )
    settings = Settings(data_dir=tmp_path / ".cognivore", prefer_semantic_embedder=False)
    return TestClient(create_app(settings))


def _upload_video(client: TestClient, name: str = "lecture.mp4", query: str = "") -> dict:
    res = client.post(
        f"/api/media/video{query}", files={"file": (name, b"not really a video", "video/mp4")}
    )
    assert res.status_code == 200, res.text
    return res.json()


def test_uploaded_video_goes_into_the_knowledge_base_with_timestamps(
    video_client: TestClient,
) -> None:
    body = _upload_video(video_client)

    assert body["source"] == "lecture.mp4"
    assert body["scenes"] == 2
    assert body["speech_language"] == "ru"
    assert body["chunks_added"] == 2
    assert body["text"].startswith("[00:00–00:10]\nНа экране: Тарифы · Старт — 490 рублей в месяц")
    assert "Речь: Все данные шифруются алгоритмом AES." in body["text"]
    assert body["analysis"] == body["text"]  # the field's earlier name still works
    chunks = video_client.get("/api/knowledge_base").json()
    assert {c["source"] for c in chunks} == {"lecture.mp4"}


def test_a_question_about_the_video_is_answered_from_it(video_client: TestClient) -> None:
    _upload_video(video_client)
    body = video_client.post("/api/chat", json={"message": "Сколько стоит тариф Старт?"}).json()

    assert "490" in body["answer"]
    assert body["insight"]["hits"][0]["source"] == "lecture.mp4"
    assert body["insight"]["hits"][0]["preview"].startswith("[00:00–00:10]")
    assert not body["insight"]["gap"]


def test_uploading_the_same_video_twice_does_not_duplicate_it(video_client: TestClient) -> None:
    _upload_video(video_client)
    again = _upload_video(video_client)

    assert again["chunks_added"] == 0
    assert "already_in_knowledge_base" in again["notes"]
    assert again["total_chunks"] == 2


def test_video_can_be_analyzed_without_adding_it(video_client: TestClient) -> None:
    body = _upload_video(video_client, query="?ingest=false")

    assert body["chunks_added"] == 0 and body["total_chunks"] == 0
    assert "Тарифы" in body["text"]


def test_uploaded_audio_goes_into_the_knowledge_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("faster_whisper", reason="the audio tool needs the `audio` extra")
    from cognivore.media.audio import Transcript, TranscriptSegment

    transcript = Transcript(
        text="",
        segments=[
            TranscriptSegment(0.0, 5.0, "Welcome to the weekly sync."),
            TranscriptSegment(70.0, 75.0, "The release moves to Friday."),
        ],
        language="en",
    )
    hints: list[str | None] = []

    def fake_transcribe(path: str, model_size: str, language_hint: str | None = None) -> object:
        hints.append(language_hint)
        return transcript

    monkeypatch.setattr("cognivore.media.audio.transcribe", fake_transcribe)
    client = TestClient(
        create_app(Settings(data_dir=tmp_path / ".cognivore", prefer_semantic_embedder=False))
    )

    res = client.post(
        "/api/media/audio?language=en", files={"file": ("sync.m4a", b"audio", "audio/mp4")}
    )
    body = res.json()

    assert hints == ["en"]  # the UI language is passed on as a hint

    assert res.status_code == 200
    assert body["chunks_added"] == 2  # two one-minute stretches of speech
    assert body["transcript"].startswith("[00:00–00:05]\nSpeech: Welcome to the weekly sync.")
    assert "[01:10–01:15]" in body["text"]


def test_audio_without_speech_says_so(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("faster_whisper", reason="the audio tool needs the `audio` extra")
    from cognivore.media.audio import Transcript

    monkeypatch.setattr(
        "cognivore.media.audio.transcribe",
        lambda path, model_size, language_hint=None: Transcript(text="", segments=[], language=""),
    )
    client = TestClient(
        create_app(Settings(data_dir=tmp_path / ".cognivore", prefer_semantic_embedder=False))
    )
    body = client.post(
        "/api/media/audio", files={"file": ("music.mp3", b"audio", "audio/mpeg")}
    ).json()

    assert body["notes"] == ["no_speech"]
    assert body["chunks_added"] == 0
