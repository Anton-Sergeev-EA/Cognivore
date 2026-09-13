"""Speech-to-text via ``faster-whisper`` (a CTranslate2 reimplementation of
OpenAI Whisper -- int8-quantized, no PyTorch, fast enough for CPU-only
machines). The dependency is optional (``pip install cognivore[audio]``);
importing this module never fails, only *using* it without the extra does.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache


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


def transcribe(audio_path: str, model_size: str = "base") -> Transcript:
    model = _load_model(model_size)
    segments_iter, info = model.transcribe(audio_path, vad_filter=True)
    segments = [
        TranscriptSegment(start=s.start, end=s.end, text=s.text.strip()) for s in segments_iter
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
