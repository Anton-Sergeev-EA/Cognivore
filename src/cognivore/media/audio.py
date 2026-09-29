"""Speech-to-text via ``faster-whisper`` (a CTranslate2 reimplementation of
OpenAI Whisper -- int8-quantized, no PyTorch, fast enough for CPU-only
machines). The dependency is optional (``pip install cognivore[audio]``);
importing this module never fails, only *using* it without the extra does.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str
    speaker: str | None = None


@dataclass
class Transcript:
    text: str
    segments: list[TranscriptSegment]
    language: str


def is_available() -> bool:
    """Whether speech recognition can run (the ``audio`` extra is installed)."""
    import importlib.util

    return importlib.util.find_spec("faster_whisper") is not None


@lru_cache(maxsize=2)
def _load_model(model_size: str):
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "faster-whisper is not installed. Install it with `pip install cognivore[audio]`."
        ) from exc
    # int8 on CPU keeps memory and latency reasonable on a laptop with no GPU.
    return WhisperModel(model_size, device="cpu", compute_type="int8")


def decode_audio(path: str, sampling_rate: int = 16000) -> Any:
    """The first audio track of ``path`` (any file FFmpeg reads: wav, mp3,
    m4a, mp4, webm, mov...) as 16 kHz mono float32 samples -- what Whisper
    takes.

    faster-whisper has a decoder of its own, but (as of 1.2.1) it passes
    PyAV an argument PyAV 18 removed, so with a current PyAV *every* file
    failed to transcribe. This one works with PyAV before and after 18.
    """
    import gc
    import io

    import av
    import numpy as np

    # Untyped on purpose: PyAV < 18 needs metadata_errors (or undecodable
    # metadata aborts the whole file), PyAV >= 18 no longer accepts it.
    open_container: Any = av.open
    try:
        container = open_container(path, mode="r", metadata_errors="ignore")
    except TypeError:
        container = open_container(path, mode="r")
    buffer = io.BytesIO()
    resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=sampling_rate)
    with container:
        if not container.streams.audio:
            return np.zeros(0, dtype=np.float32)
        for frame in container.decode(audio=0):
            try:
                resampled = resampler.resample(frame)
            except av.error.InvalidDataError:  # a damaged frame: skip it
                continue
            for chunk in resampled:
                buffer.write(chunk.to_ndarray().tobytes())
        for chunk in resampler.resample(None):  # flush
            buffer.write(chunk.to_ndarray().tobytes())
    # PyAV's resampler holds native buffers until collected (the same
    # workaround faster-whisper uses).
    del resampler
    gc.collect()
    return np.frombuffer(buffer.getbuffer(), dtype=np.int16).astype(np.float32) / 32768.0


# Below this, Whisper's own language guess is not trusted over a hint
# (measured: robotic Russian speech was guessed as Georgian at 0.31).
_TRUSTED_LANGUAGE_PROBABILITY = 0.5
# Whisper's own defaults for "this segment is real speech": segments below
# this average log-probability or above this no-speech probability are
# dropped when transcribing without voice-activity detection (measured:
# gibberish -2.09, real speech -0.5 to -0.8).
_MIN_AVG_LOGPROB = -1.0
_MAX_NO_SPEECH_PROB = 0.6


def transcribe(
    audio_path: str, model_size: str = "small", language_hint: str | None = None
) -> Transcript:
    """Speech of ``audio_path`` as timed segments.

    ``language_hint`` (an ISO code such as ``"ru"`` -- the language of a
    video's on-screen text, or of the UI) is used when Whisper isn't sure
    which language it hears: its guess on short or unusual speech can be
    wildly off, and a wrong language turns the whole transcript into
    gibberish. When voice-activity detection finds no speech at all, the
    file is transcribed once more without it, keeping only segments Whisper
    itself is confident in.
    """
    model = _load_model(model_size)
    audio = decode_audio(audio_path)
    if not audio.size:
        return Transcript(text="", segments=[], language="")

    def run(vad: bool, language: str | None) -> tuple[list[Any], Any]:
        segments_iter, info = model.transcribe(audio, vad_filter=vad, language=language)
        if (
            language is None
            and language_hint
            and info.language != language_hint
            and info.language_probability < _TRUSTED_LANGUAGE_PROBABILITY
        ):
            # Segments are generated lazily, so nothing was decoded yet.
            return run(vad, language_hint)
        return list(segments_iter), info

    raw, info = run(vad=True, language=None)
    if not raw:
        raw, info = run(vad=False, language=None)
        raw = [
            s
            for s in raw
            if s.avg_logprob >= _MIN_AVG_LOGPROB and s.no_speech_prob <= _MAX_NO_SPEECH_PROB
        ]
    segments = [
        TranscriptSegment(start=s.start, end=s.end, text=s.text.strip())
        for s in raw
        if s.text.strip()
    ]
    full_text = " ".join(s.text for s in segments).strip()
    return Transcript(text=full_text, segments=segments, language=info.language)


def diarize_by_energy(
    segments: list[TranscriptSegment], max_speakers: int = 2
) -> list[TranscriptSegment]:
    """A deliberately simple speaker-turn heuristic: alternates speaker
    labels on sufficiently long silence gaps between segments. This is
    *not* real diarization (that needs a speaker-embedding model), but it's
    a genuinely useful, dependency-free approximation for two-person
    conversations, and the interface is where a real pyannote-based
    diarizer would plug in later -- see the roadmap in README.md.
    """
    if not segments:
        return []
    labeled: list[TranscriptSegment] = []
    current_speaker = 0
    prev_end = segments[0].start
    gap_threshold = 1.2  # seconds of silence treated as a likely turn change
    for seg in segments:
        if seg.start - prev_end > gap_threshold:
            current_speaker = (current_speaker + 1) % max_speakers
        labeled.append(
            TranscriptSegment(
                start=seg.start, end=seg.end, text=seg.text, speaker=f"speaker_{current_speaker}"
            )
        )
        prev_end = seg.end
    return labeled
