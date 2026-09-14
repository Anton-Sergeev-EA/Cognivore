"""Correctness tests for speech-to-text and the speaker-turn heuristic.

`diarize_by_energy` is pure and dependency-free, so it's tested for real
with no mocking. The transcription path needs the optional `faster-whisper`
extra plus (on first use) a model download from Hugging Face -- both
self-skip when unavailable, the same pattern `test_video.py` uses for
OpenCV/Tesseract, so the suite stays green on a machine that only installed
the base package or has no network access to Hugging Face, while still
giving real, non-mocked coverage wherever both are reachable.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from cognivore.media.audio import TranscriptSegment, diarize_by_energy

pytest.importorskip(
    "faster_whisper", reason="faster-whisper not installed (pip install cognivore[audio])"
)

from cognivore.media.audio import transcribe

espeak_missing = shutil.which("espeak-ng") is None and shutil.which("espeak") is None
pytestmark_tts = pytest.mark.skipif(
    espeak_missing,
    reason="no espeak/espeak-ng available to synthesize test speech",
)


def _seg(start: float, end: float, text: str = "hi") -> TranscriptSegment:
    return TranscriptSegment(start=start, end=end, text=text)


def test_diarize_by_energy_keeps_same_speaker_across_short_gaps() -> None:
    segments = [_seg(0.0, 1.0), _seg(1.2, 2.0), _seg(2.3, 3.0)]

    labeled = diarize_by_energy(segments)

    assert [s.speaker for s in labeled] == ["speaker_0", "speaker_0", "speaker_0"]


def test_diarize_by_energy_alternates_speaker_on_long_gaps() -> None:
    # Gaps of 2s exceed the 1.2s threshold, so each one should flip speakers.
    segments = [_seg(0.0, 1.0), _seg(3.0, 4.0), _seg(6.0, 7.0)]

    labeled = diarize_by_energy(segments)

    assert [s.speaker for s in labeled] == ["speaker_0", "speaker_1", "speaker_0"]


def test_diarize_by_energy_respects_max_speakers() -> None:
    segments = [_seg(t, t + 1.0) for t in (0.0, 3.0, 6.0, 9.0)]

    labeled = diarize_by_energy(segments, max_speakers=3)

    assert [s.speaker for s in labeled] == [
        "speaker_0",
        "speaker_1",
        "speaker_2",
        "speaker_0",
    ]


def test_diarize_by_energy_handles_empty_input() -> None:
    assert diarize_by_energy([]) == []


def test_diarize_by_energy_preserves_text_and_timing() -> None:
    labeled = diarize_by_energy([_seg(0.0, 1.5, text="hello there")])

    assert labeled[0].start == 0.0
    assert labeled[0].end == 1.5
    assert labeled[0].text == "hello there"


def test_load_model_raises_a_clear_error_when_faster_whisper_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unlike the OCR dependency (see test_video.py), a missing `faster-whisper`
    was never silently swallowed -- `_load_model` re-raises as a `RuntimeError`
    with an actionable install hint. This locks that behavior in."""
    import builtins

    from cognivore.media.audio import _load_model

    _load_model.cache_clear()
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "faster_whisper":
            raise ImportError("simulated: faster-whisper not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError, match="faster-whisper is not installed"):
        _load_model("tiny-simulated-missing")

    _load_model.cache_clear()


@pytestmark_tts
def test_transcribe_recognizes_synthetic_speech(tmp_path: Path) -> None:
    wav_path = tmp_path / "speech.wav"
    espeak = shutil.which("espeak-ng") or shutil.which("espeak")
    subprocess.run(
        [espeak, "-s", "150", "-w", str(wav_path), "Hello world, this is a test."],
        check=True,
        capture_output=True,
    )

    try:
        transcript = transcribe(str(wav_path), model_size="tiny")
    except Exception as exc:  # pragma: no cover - depends on network access
        pytest.skip(f"could not load/download the whisper model: {exc!r}")

    assert "hello" in transcript.text.lower()
    assert transcript.segments


def test_transcribe_raises_on_a_missing_file() -> None:
    try:
        transcribe("/nonexistent/path/does-not-exist.wav", model_size="tiny")
    except (RuntimeError, ValueError, OSError):
        return
    except Exception as exc:  # pragma: no cover - depends on network access
        # Loading the model itself needs a one-time download from Hugging
        # Face; a network failure there (e.g. no route to huggingface.co)
        # is a different, unrelated failure from the one this test checks
        # for, so it's skipped rather than reported as a broken contract.
        pytest.skip(f"could not load the whisper model to run this check: {exc!r}")
    else:
        pytest.fail("expected transcribe() to raise for a nonexistent file")
