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


def _whisper_or_skip(model_size: str = "tiny") -> None:
    """Loads the model once, skipping the test when it can't be downloaded
    (no network access to Hugging Face). Only *that* is a reason to skip:
    everything after it -- decoding, transcribing -- must work or fail.
    (A catch-all skip here once hid a real bug: with PyAV 18 every file
    failed to decode, and the tests reported it as "no model".)"""
    from cognivore.media.audio import _load_model

    try:
        _load_model(model_size)
    except Exception as exc:  # pragma: no cover - depends on network access
        pytest.skip(f"could not load/download the whisper model: {exc!r}")


def _write_tone(path: Path, seconds: float = 1.0, rate: int = 44100) -> None:
    """A stereo sine tone as a 16-bit WAV, written with the standard library."""
    import math
    import struct
    import wave

    frames = bytearray()
    for i in range(int(seconds * rate)):
        sample = int(12000 * math.sin(2 * math.pi * 440 * i / rate))
        frames += struct.pack("<hh", sample, sample)
    with wave.open(str(path), "wb") as out:
        out.setnchannels(2)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(bytes(frames))


def test_decode_audio_gives_16khz_mono_samples(tmp_path: Path) -> None:
    from cognivore.media.audio import decode_audio

    path = tmp_path / "tone.wav"
    _write_tone(path, seconds=1.0)
    audio = decode_audio(str(path))

    assert audio.dtype.name == "float32"
    assert abs(audio.size - 16000) < 200
    assert 0.3 < float(abs(audio).max()) < 0.4  # 12000 / 32768


def test_decode_audio_of_a_file_without_sound_is_empty(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    path = tmp_path / "silent.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 64))
    for _ in range(10):
        writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
    writer.release()

    from cognivore.media.audio import decode_audio

    assert decode_audio(str(path)).size == 0


def test_transcribe_hands_whisper_decoded_samples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """transcribe() decodes the file itself and gives Whisper the samples
    (see decode_audio for why), and maps Whisper's segments to ours."""
    import numpy as np

    from cognivore.media import audio as audio_module

    class FakeSegment:
        def __init__(self, start: float, end: float, text: str) -> None:
            self.start, self.end, self.text = start, end, text

    class FakeInfo:
        language = "en"
        language_probability = 0.95

    class FakeModel:
        def transcribe(
            self, audio: object, vad_filter: bool, language: str | None
        ) -> tuple[object, FakeInfo]:
            assert isinstance(audio, np.ndarray) and audio.size > 0
            return iter([FakeSegment(0.0, 0.8, " Hello there. ")]), FakeInfo()

    monkeypatch.setattr(audio_module, "_load_model", lambda size: FakeModel())
    path = tmp_path / "tone.wav"
    _write_tone(path)

    transcript = audio_module.transcribe(str(path))

    assert transcript.language == "en"
    assert transcript.text == "Hello there."
    assert [(s.start, s.end) for s in transcript.segments] == [(0.0, 0.8)]


class _Seg:
    def __init__(self, text: str, logprob: float = -0.5, no_speech: float = 0.1) -> None:
        self.start, self.end, self.text = 0.0, 1.0, text
        self.avg_logprob, self.no_speech_prob = logprob, no_speech


class _Info:
    def __init__(self, language: str, probability: float) -> None:
        self.language, self.language_probability = language, probability


class _ScriptedModel:
    """A stand-in for WhisperModel that answers from a table keyed by
    (vad_filter, language) and records the calls -- the numbers are the
    ones measured on a real run."""

    def __init__(self, answers: dict[tuple[bool, str | None], tuple[list[_Seg], _Info]]) -> None:
        self.answers = answers
        self.calls: list[tuple[bool, str | None]] = []

    def transcribe(self, audio: object, vad_filter: bool, language: str | None) -> tuple:
        self.calls.append((vad_filter, language))
        segments, info = self.answers[(vad_filter, language)]
        return iter(segments), info


def _transcribe_with(
    model: _ScriptedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **kwargs: object
) -> object:
    from cognivore.media import audio as audio_module

    monkeypatch.setattr(audio_module, "_load_model", lambda size: model)
    path = tmp_path / "tone.wav"
    _write_tone(path)
    return audio_module.transcribe(str(path), **kwargs)


def test_an_unsure_language_guess_gives_way_to_the_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _ScriptedModel(
        {
            (True, None): ([_Seg("sivabh mātāyat")], _Info("ka", 0.31)),
            (True, "ru"): ([_Seg("Добрый день.")], _Info("ru", 1.0)),
        }
    )
    transcript = _transcribe_with(model, tmp_path, monkeypatch, language_hint="ru")

    assert model.calls == [(True, None), (True, "ru")]
    assert transcript.text == "Добрый день." and transcript.language == "ru"


def test_a_confident_language_guess_is_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _ScriptedModel({(True, None): ([_Seg("Guten Tag.")], _Info("de", 0.97))})
    transcript = _transcribe_with(model, tmp_path, monkeypatch, language_hint="ru")

    assert model.calls == [(True, None)]
    assert transcript.language == "de"


def test_when_vad_hears_nothing_only_confident_segments_are_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _ScriptedModel(
        {
            (True, "ru"): ([], _Info("ru", 1.0)),
            (True, None): ([], _Info("ru", 0.9)),
            (False, None): (
                [
                    _Seg("Добрый день.", logprob=-0.52),
                    _Seg("sivabh mātāyat", logprob=-2.09),  # gibberish
                    _Seg("[музыка]", logprob=-0.4, no_speech=0.9),  # not speech
                ],
                _Info("ru", 0.9),
            ),
        }
    )
    transcript = _transcribe_with(model, tmp_path, monkeypatch)

    assert model.calls == [(True, None), (False, None)]
    assert [s.text for s in transcript.segments] == ["Добрый день."]


@pytestmark_tts
def test_transcribe_recognizes_synthetic_speech(tmp_path: Path) -> None:
    wav_path = tmp_path / "speech.wav"
    espeak = shutil.which("espeak-ng") or shutil.which("espeak")
    subprocess.run(
        [espeak, "-s", "150", "-w", str(wav_path), "Hello world, this is a test."],
        check=True,
        capture_output=True,
    )
    _whisper_or_skip("tiny")

    transcript = transcribe(str(wav_path), model_size="tiny")

    assert "hello" in transcript.text.lower()
    assert transcript.segments


def test_transcribe_raises_on_a_missing_file() -> None:
    _whisper_or_skip("tiny")

    with pytest.raises(OSError):
        transcribe("/nonexistent/path/does-not-exist.wav", model_size="tiny")
