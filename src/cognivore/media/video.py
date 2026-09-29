"""Lightweight, CPU-only video understanding.

Rather than reaching for a vision-language model (PyTorch plus a multi-GB
checkpoint -- a poor fit for "runs on a CPU-only laptop"), this module
extracts what is cheap to get and is usually what a video is *about* --
for lectures, tutorials, meeting recordings, screencasts and slide decks:

* **what is on screen**: every distinct scene (a slide, a screen), found by
  comparing frames, and the text on it (:mod:`cognivore.media.ocr`);
* **what is said**: the speech of the audio track, transcribed by
  :mod:`cognivore.media.audio` (``faster-whisper``), when the ``audio``
  extra is installed.

The two are merged into one timeline -- each scene with its time range, its
on-screen text and what was said while it was shown -- which the web UI
adds to the knowledge base, so the video can be asked about like any other
document, with timestamps in the cited passages.

Scene detection samples a couple of frames per second and compares small
grayscale copies pixel by pixel. That catches a slide change on an
unchanged background (the case a histogram comparison misses: the same
white slide with different text has almost the same histogram), waits for
a transition to settle before taking the frame (crossfades, build-up
animations), and for continuously moving footage takes a frame every few
seconds instead of on every change.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from itertools import pairwise
from typing import Any

from cognivore.media.ocr import OcrEngine

logger = logging.getLogger(__name__)

# Frames are compared as 480x270 grayscale: enough to see a single changed
# number on a slide, small enough to be cheap.
_THUMB = (480, 270)
# Share of thumbnail pixels that must change for "something new is on
# screen" (measured: two template slides differing in one number change
# 0.1-0.15% of pixels; an unchanged slide, 0%)...
_MIN_CHANGE = 0.0005
# ...and below which two consecutive samples count as "the picture settled"
# (moving camera footage: 0.5-0.8%).
_STABLE = 0.0003
# Moving footage never settles; take a frame at least this often anyway.
_MAX_UNSETTLED_SECONDS = 8.0
# Two scenes whose text is at least this similar are the same slide.
_SAME_TEXT = 0.9


@dataclass
class Keyframe:
    """One scene: when it appears (``timestamp``) and ends (``end``), in
    seconds, and the text on it."""

    timestamp: float
    frame_index: int
    ocr_text: str = ""
    end: float | None = None


@dataclass
class VideoAnalysis:
    duration: float
    scenes: list[Keyframe]
    # The speech of the audio track (a cognivore.media.audio.Transcript),
    # or None when there is none or it couldn't be transcribed.
    transcript: Any = None
    ocr_languages: str = ""
    # Machine-readable reasons part of the analysis is missing, for the UI
    # to explain: "ocr_unavailable", "ocr_languages_missing",
    # "speech_unavailable", "speech_failed", "no_speech", "scenes_truncated".
    notes: list[str] = field(default_factory=list)


def _require_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "opencv-python-headless is not installed. Install it with `pip install cognivore[video]`."
        ) from exc
    return cv2


def _changed(cv2: Any, a: Any, b: Any, threshold: float) -> float:
    """Share of pixels whose brightness differs by more than ``threshold``."""
    diff = cv2.absdiff(a, b)
    return float((diff > threshold).mean())


def _numbers(text: str) -> list[str]:
    return re.findall(r"\d+(?:[.,]\d+)?", text)


def _is_update_of(old: str, new: str) -> bool:
    """``new`` shows the same slide as ``old``: near-identical text (OCR
    reads the same slide slightly differently from frame to frame), or
    ``old`` plus more (a bullet list revealed one line at a time). Slides
    made from one template ("Section 1", "Section 2") differ only in their
    numbers, so different numbers always mean a different slide."""
    if not old or not new:
        return old == new
    if _numbers(old) == _numbers(new) and SequenceMatcher(None, old, new).ratio() >= _SAME_TEXT:
        return True
    old_lines = [ln for ln in old.splitlines() if ln.strip()]
    kept = sum(1 for ln in old_lines if ln in new)
    return len(new) > len(old) and kept >= max(1, round(0.9 * len(old_lines)))


def extract_keyframes(
    video_path: str,
    scene_threshold: float = 30.0,
    max_keyframes: int = 200,
    run_ocr: bool = True,
    ocr_languages: str = "auto",
    sample_every: float = 0.5,
    ocr: OcrEngine | None = None,
    notes: list[str] | None = None,
) -> list[Keyframe]:
    """The distinct scenes of a video, each with its on-screen text.

    ``scene_threshold`` is how much (0-255) a pixel's brightness must change
    to count as changed. ``ocr`` lets a caller share one engine (and learn
    which languages it used); ``notes`` collects the reasons anything was
    skipped (see :class:`VideoAnalysis`).
    """
    cv2 = _require_cv2()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"could not open video: {video_path}")
    if notes is None:
        notes = []
    if run_ocr and ocr is None:
        ocr = OcrEngine(ocr_languages)
    if run_ocr and ocr is not None:
        if not ocr.available:
            notes.append("ocr_unavailable")
        elif ocr.missing:
            notes.append("ocr_languages_missing")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if not 1.0 <= fps <= 240.0:  # some containers report nonsense
        fps = 25.0
    step = max(1, round(fps * sample_every))

    scenes: list[Keyframe] = []
    reference = previous = None
    previous_frame = None
    previous_index = 0
    pending_since: float | None = None
    frame_index = -1

    reading = run_ocr and ocr is not None and ocr.available

    def take(frame: Any, index: int) -> None:
        text = ocr.read(frame) if reading and ocr is not None else ""
        timestamp = index / fps
        # Merging by text needs text: without OCR every scene found by the
        # pixel comparison is kept.
        if reading and scenes and _is_update_of(scenes[-1].ocr_text, text):
            # The same slide again (or with more of its bullets revealed):
            # keep one scene, with the fullest text.
            if len(text) > len(scenes[-1].ocr_text):
                scenes[-1].ocr_text = text
            return
        scenes.append(Keyframe(timestamp=timestamp, frame_index=index, ocr_text=text))

    try:
        while True:
            frame_index += 1
            if frame_index % step:
                if not cap.grab():
                    break
                continue
            ok, frame = cap.read()
            if not ok:
                break
            now = frame_index / fps
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            thumb = cv2.GaussianBlur(
                cv2.resize(gray, _THUMB, interpolation=cv2.INTER_AREA), (3, 3), 0
            )

            if reference is None or _changed(cv2, thumb, reference, scene_threshold) > _MIN_CHANGE:
                if pending_since is None:
                    pending_since = now
                settled = previous is not None and (
                    _changed(cv2, thumb, previous, scene_threshold) < _STABLE
                )
                if settled and previous_frame is not None:
                    take(previous_frame, previous_index)
                    reference, pending_since = thumb, None
                elif now - pending_since >= _MAX_UNSETTLED_SECONDS:
                    take(frame, frame_index)
                    reference, pending_since = thumb, None
            else:
                pending_since = None

            previous, previous_frame, previous_index = thumb, frame, frame_index
            if len(scenes) >= max_keyframes:
                notes.append("scenes_truncated")
                break

        # A last scene still settling when the video ended (e.g. a slide
        # shown for less than a sample interval) still counts.
        if pending_since is not None and previous_frame is not None and len(scenes) < max_keyframes:
            take(previous_frame, previous_index)
        duration = frame_index / fps
    finally:
        cap.release()

    for current, following in pairwise(scenes):
        current.end = following.timestamp
    if scenes:
        scenes[-1].end = max(duration, scenes[-1].timestamp)
    return scenes


def _has_audio_track(video_path: str) -> bool | None:
    """Whether the file has an audio stream; ``None`` if that can't be told
    (PyAV, which faster-whisper installs, isn't available)."""
    try:
        import av
    except ImportError:
        return None
    try:
        with av.open(video_path) as container:
            return bool(container.streams.audio)
    except Exception:
        return None


def analyze_video(
    video_path: str,
    scene_threshold: float = 30.0,
    max_keyframes: int = 200,
    ocr_languages: str = "auto",
    transcribe_speech: bool = True,
    whisper_model_size: str = "small",
    transcriber: Callable[[str, str, str | None], Any] | None = None,
    language_hint: str | None = None,
) -> VideoAnalysis:
    """Scenes with their on-screen text, plus the speech of the audio track.

    ``transcriber(path, model_size, language_hint)`` defaults to
    :func:`cognivore.media.audio.transcribe`; any failure there (no
    ``audio`` extra, no model download possible) leaves the on-screen part
    intact and is reported in ``notes``. The language of the on-screen text
    -- or else ``language_hint`` (e.g. the UI language) -- tells Whisper
    what to listen for when it can't tell by itself.
    """
    notes: list[str] = []
    ocr = OcrEngine(ocr_languages)
    scenes = extract_keyframes(
        video_path,
        scene_threshold=scene_threshold,
        max_keyframes=max_keyframes,
        ocr_languages=ocr_languages,
        ocr=ocr,
        notes=notes,
    )
    duration = (scenes[-1].end or 0.0) if scenes else 0.0

    transcript = None
    has_audio = _has_audio_track(video_path)
    if transcribe_speech and has_audio is not False:
        from cognivore.ml.lang import detect_language_confident

        on_screen = " ".join(s.ocr_text for s in scenes)
        hint = detect_language_confident(on_screen, min_evidence=2) or language_hint
        if transcriber is None:
            from cognivore.media import audio

            if audio.is_available():
                transcriber = audio.transcribe
            else:
                notes.append("speech_unavailable")
        if transcriber is not None:
            try:
                transcript = transcriber(video_path, whisper_model_size, hint)
            except Exception:
                logger.warning("Could not transcribe the audio track.", exc_info=True)
                notes.append("speech_failed")
            else:
                if transcript is not None and transcript.segments:
                    duration = max(duration, transcript.segments[-1].end)
                    if scenes and scenes[-1].end is not None:
                        scenes[-1].end = max(scenes[-1].end, duration)
                else:
                    transcript = None
                    if has_audio:
                        notes.append("no_speech")

    return VideoAnalysis(
        duration=duration,
        scenes=scenes,
        transcript=transcript,
        ocr_languages=ocr.languages_used if ocr.available else "",
        notes=notes,
    )


# -- Presentation --------------------------------------------------------

# Section labels in the language of the video's content.
_LABELS = {
    "ru": ("На экране", "Речь"),
    "en": ("On screen", "Speech"),
    "zh": ("屏幕内容", "语音"),
    "es": ("En pantalla", "Voz"),
    "hi": ("स्क्रीन पर", "वाणी"),
    "fr": ("À l'écran", "Parole"),
    "de": ("Auf dem Bildschirm", "Sprache"),
    "ja": ("画面", "音声"),
    "it": ("Sullo schermo", "Parlato"),
}
# Speech with no on-screen scenes to hang it on is cut into sections this long.
_SPEECH_SECTION_SECONDS = 60.0


def _clock(seconds: float) -> str:
    total = int(seconds)
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


@dataclass
class TimelineSection:
    start: float
    end: float
    on_screen: str
    speech: str


def build_timeline(analysis: VideoAnalysis) -> list[TimelineSection]:
    """Scenes with on-screen text, each with the speech heard while it was
    shown; speech over text-less footage (a talking head, a camera shot) is
    cut into sections of its own, about a minute each."""
    segments = list(analysis.transcript.segments) if analysis.transcript else []
    sections: list[TimelineSection] = []
    used = [False] * len(segments)
    for scene in analysis.scenes:
        if not scene.ocr_text:
            continue
        end = scene.end if scene.end is not None else scene.timestamp
        speech = []
        for i, seg in enumerate(segments):
            midpoint = (seg.start + seg.end) / 2
            if not used[i] and scene.timestamp <= midpoint < max(end, scene.timestamp + 1e-6):
                speech.append(seg.text)
                used[i] = True
        sections.append(TimelineSection(scene.timestamp, end, scene.ocr_text, " ".join(speech)))

    # Speech that fell outside every scene with text.
    window: list[Any] = []
    for i, seg in enumerate(segments):
        if used[i]:
            continue
        if window and seg.end - window[0].start > _SPEECH_SECTION_SECONDS:
            sections.append(
                TimelineSection(
                    window[0].start, window[-1].end, "", " ".join(s.text for s in window)
                )
            )
            window = []
        window.append(seg)
    if window:
        sections.append(
            TimelineSection(window[0].start, window[-1].end, "", " ".join(s.text for s in window))
        )
    sections.sort(key=lambda s: s.start)
    return sections


def _content_language(analysis: VideoAnalysis, sections: list[TimelineSection]) -> str:
    """The language for the section labels: the one Whisper heard, if there
    was speech (it is reliable); otherwise the one of the on-screen text,
    English unless the text clearly says otherwise."""
    from cognivore.ml.lang import detect_language_confident

    if analysis.transcript is not None and analysis.transcript.language in _LABELS:
        return str(analysis.transcript.language)
    sample = " ".join(f"{s.on_screen} {s.speech}" for s in sections)[:2000]
    return detect_language_confident(sample, min_evidence=2) or "en"


def section_text(section: TimelineSection, language: str) -> str:
    """One timeline section as a self-contained passage: a knowledge-base
    chunk made from it still says when in the video it is from."""
    on_screen_label, speech_label = _LABELS.get(language, _LABELS["en"])
    lines = [f"[{_clock(section.start)}–{_clock(section.end)}]"]
    if section.on_screen:
        lines.append(f"{on_screen_label}: " + " · ".join(section.on_screen.splitlines()))
    if section.speech:
        lines.append(f"{speech_label}: {section.speech}")
    return "\n".join(lines)


def timeline_passages(analysis: VideoAnalysis) -> list[str]:
    sections = build_timeline(analysis)
    language = _content_language(analysis, sections)
    return [section_text(s, language) for s in sections]


def summarize_analysis(analysis: VideoAnalysis) -> str:
    """The whole timeline as text (what the agent's tool returns)."""
    passages = timeline_passages(analysis)
    if not passages:
        if analysis.scenes:
            return "No on-screen text or speech was found in this video."
        return "No distinct scenes were detected."
    return "\n\n".join(passages)


def summarize_keyframes(keyframes: list[Keyframe]) -> str:
    """Scenes alone as text (kept for callers that only extract keyframes)."""
    if not keyframes:
        return "No distinct scenes were detected."
    return summarize_analysis(VideoAnalysis(duration=keyframes[-1].end or 0.0, scenes=keyframes))
