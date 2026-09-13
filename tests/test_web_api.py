from __future__ import annotations

import importlib.util
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
    for path in ("/", "/app.js", "/i18n.js", "/styles.css"):
        res = client.get(path)
        assert res.headers.get("cache-control") == "no-store", path
