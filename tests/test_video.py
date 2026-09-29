"""Video understanding, tested on synthetic videos built the way real ones
look: slides with the same background and only the text changing,
crossfades, bullets revealed one at a time, template slides that differ
only in a number, moving camera footage.

OpenCV is optional (``cognivore[video]``) and OCR also needs the Tesseract
binary, so these tests self-skip when either is missing (the CI installs
both on one leg of the matrix; the Docker image ships both). The timeline
tests at the end need neither.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest

from cognivore.media.audio import Transcript, TranscriptSegment
from cognivore.media.video import (
    Keyframe,
    VideoAnalysis,
    build_timeline,
    section_text,
    summarize_analysis,
    timeline_passages,
)

W, H, FPS = 640, 360, 10
_FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")


def _has_tesseract_pack(pack: str) -> bool:
    if shutil.which("tesseract") is None:
        return False
    try:
        import pytesseract

        return pack in pytesseract.get_languages(config="")
    except Exception:
        return False


needs_cv2 = pytest.mark.skipif(
    __import__("importlib").util.find_spec("cv2") is None,
    reason="opencv-python-headless not installed (pip install cognivore[video])",
)
needs_ocr = pytest.mark.skipif(
    not _has_tesseract_pack("eng") or not (_FONT_DIR / "DejaVuSans.ttf").exists(),
    reason="tesseract (with eng) or the DejaVu font is not installed",
)
needs_ocr_rus = pytest.mark.skipif(
    not _has_tesseract_pack("rus"), reason="tesseract-ocr-rus is not installed"
)


def _slide(title: str, lines: list[str] = (), background: int = 250) -> Any:  # type: ignore[assignment]
    """A slide as a BGR frame, drawn with a Unicode font (cv2.putText is
    ASCII-only)."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (W, H), (background,) * 3)
    draw = ImageDraw.Draw(img)
    bold = ImageFont.truetype(str(_FONT_DIR / "DejaVuSans-Bold.ttf"), 34)
    regular = ImageFont.truetype(str(_FONT_DIR / "DejaVuSans.ttf"), 26)
    draw.text((40, 40), title, font=bold, fill=(10, 10, 10))
    for i, line in enumerate(lines):
        draw.text((50, 120 + i * 50), line, font=regular, fill=(10, 10, 10))
    return np.array(img)[:, :, ::-1].copy()


def _write(path: Path, frames: list[Any]) -> str:
    import cv2

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()
    return str(path)


def _hold(frame: Any, seconds: float) -> list[Any]:
    return [frame] * int(seconds * FPS)


def _crossfade(a: Any, b: Any, seconds: float = 1.0) -> list[Any]:
    import cv2

    n = int(seconds * FPS)
    return [cv2.addWeighted(a, 1 - (k + 1) / n, b, (k + 1) / n, 0) for k in range(n)]


def _scenes(path: str, **kwargs: Any) -> list[Keyframe]:
    from cognivore.media.video import extract_keyframes

    return extract_keyframes(path, **kwargs)


# -- Scene detection (OpenCV only) --------------------------------------


@needs_cv2
def test_slides_on_the_same_background_are_separate_scenes(tmp_path: Path) -> None:
    """The case the old histogram comparison missed: a white slide with
    other text has almost the same brightness histogram."""
    if not (_FONT_DIR / "DejaVuSans.ttf").exists():
        pytest.skip("DejaVu font not installed")
    frames = []
    for i in range(3):
        frames += _hold(_slide(f"Slide {i + 1}", [f"Point number {i + 1}"]), 2)
    scenes = _scenes(_write(tmp_path / "v.mp4", frames), run_ocr=False)

    assert len(scenes) == 3
    for i, scene in enumerate(scenes):
        assert abs(scene.timestamp - 2 * i) <= 0.6
    assert scenes[-1].end == pytest.approx(6.0, abs=0.2)


@needs_cv2
def test_crossfades_give_one_scene_per_slide(tmp_path: Path) -> None:
    if not (_FONT_DIR / "DejaVuSans.ttf").exists():
        pytest.skip("DejaVu font not installed")
    slides = [_slide(f"Topic {i}", ["Some text", f"Detail {i}"]) for i in range(3)]
    frames = _hold(slides[0], 2) + _crossfade(slides[0], slides[1]) + _hold(slides[1], 2)
    frames += _crossfade(slides[1], slides[2]) + _hold(slides[2], 2)

    assert len(_scenes(_write(tmp_path / "v.mp4", frames), run_ocr=False)) == 3


@needs_cv2
def test_a_static_video_is_one_scene(tmp_path: Path) -> None:
    import numpy as np

    frames = _hold(np.full((H, W, 3), 128, dtype=np.uint8), 5)
    assert len(_scenes(_write(tmp_path / "v.mp4", frames), run_ocr=False)) == 1


@needs_cv2
def test_moving_footage_takes_a_frame_every_few_seconds(tmp_path: Path) -> None:
    """Camera footage never settles: a frame every ~8 s, not one per sample."""
    import numpy as np

    rng = np.random.default_rng(0)
    xx = np.tile(np.arange(W, dtype=np.float32), (H, 1))
    frames = []
    for f in range(20 * FPS):
        base = (xx + f * 12) % W / W * 150 + 50
        noise = rng.normal(0, 10, (H, W))
        frames.append(np.clip(base + noise, 0, 255).astype(np.uint8)[:, :, None].repeat(3, 2))
    scenes = _scenes(_write(tmp_path / "v.mp4", frames), run_ocr=False)

    assert 1 <= len(scenes) <= 4


@needs_cv2
def test_max_keyframes_stops_and_says_so(tmp_path: Path) -> None:
    if not (_FONT_DIR / "DejaVuSans.ttf").exists():
        pytest.skip("DejaVu font not installed")
    frames = []
    for i in range(4):
        frames += _hold(_slide(f"Slide {i + 1}"), 1.5)
    notes: list[str] = []
    scenes = _scenes(
        _write(tmp_path / "v.mp4", frames), run_ocr=False, max_keyframes=2, notes=notes
    )

    assert len(scenes) == 2
    assert "scenes_truncated" in notes


@needs_cv2
def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        _scenes(str(tmp_path / "does-not-exist.mp4"))


# -- With OCR -----------------------------------------------------------


@needs_cv2
@needs_ocr
def test_slide_text_is_read_and_cleaned(tmp_path: Path) -> None:
    frames = _hold(_slide("Pricing", ["• Starter: $9 per month", "• Team: $29 per user"]), 2)
    frames += _hold(_slide("Support", ["• Response within 4 hours"]), 2)
    scenes = _scenes(_write(tmp_path / "v.mp4", frames), ocr_languages="eng")

    assert [s.ocr_text.splitlines()[0] for s in scenes] == ["Pricing", "Support"]
    assert "Starter: $9 per month" in scenes[0].ocr_text
    assert "•" not in scenes[0].ocr_text


@needs_cv2
@needs_ocr
def test_template_slides_differing_in_a_number_are_all_kept(tmp_path: Path) -> None:
    frames = []
    for i in range(5):
        frames += _hold(_slide(f"Section {i + 1}", [f"Metric: {100 + 7 * i} units"]), 1.5)
    scenes = _scenes(_write(tmp_path / "v.mp4", frames), ocr_languages="eng")

    assert [s.ocr_text.splitlines()[0] for s in scenes] == [f"Section {i}" for i in range(1, 6)]


@needs_cv2
@needs_ocr
def test_bullets_revealed_one_by_one_make_one_scene(tmp_path: Path) -> None:
    bullets = ["First step: sign up", "Second step: pick a plan", "Third step: pay"]
    frames = []
    for k in range(1, 4):
        frames += _hold(_slide("Getting started", bullets[:k]), 1.5)
    frames += _hold(_slide("Export", ["Open Settings"]), 1.5)
    scenes = _scenes(_write(tmp_path / "v.mp4", frames), ocr_languages="eng")

    assert len(scenes) == 2
    assert all(b in scenes[0].ocr_text for b in bullets)
    assert scenes[0].timestamp == pytest.approx(0.0, abs=0.6)


@needs_cv2
@needs_ocr
@needs_ocr_rus
def test_cyrillic_slides_are_read_with_the_right_language(tmp_path: Path) -> None:
    frames = _hold(_slide("Политика возврата", ["Полный возврат в течение 14 дней"]), 2)
    scenes = _scenes(_write(tmp_path / "v.mp4", frames), ocr_languages="auto")

    assert "Политика возврата" in scenes[0].ocr_text
    assert "14 дней" in scenes[0].ocr_text


@needs_cv2
@needs_ocr
def test_missing_language_pack_is_reported(tmp_path: Path) -> None:
    frames = _hold(_slide("Hello"), 1)
    notes: list[str] = []
    _scenes(_write(tmp_path / "v.mp4", frames), ocr_languages="eng+not_a_pack", notes=notes)

    assert notes == ["ocr_languages_missing"]


@needs_cv2
@needs_ocr
def test_analyze_video_merges_speech_into_the_scenes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cognivore.media.video import analyze_video

    frames = _hold(_slide("Pricing", ["Starter: $9 per month"]), 3)
    frames += _hold(_slide("Support", ["Response within 4 hours"]), 3)
    path = _write(tmp_path / "v.mp4", frames)

    def fake_transcriber(video_path: str, model_size: str, hint: str | None) -> Transcript:
        assert video_path == path
        return Transcript(
            text="",
            segments=[
                TranscriptSegment(0.5, 2.0, "The starter plan is nine dollars."),
                TranscriptSegment(3.5, 5.0, "We answer within four hours."),
            ],
            language="en",
        )

    # Videos written by OpenCV have no audio track; pretend this one has.
    monkeypatch.setattr("cognivore.media.video._has_audio_track", lambda _: True)
    analysis = analyze_video(path, ocr_languages="eng", transcriber=fake_transcriber)
    passages = timeline_passages(analysis)

    assert len(passages) == 2
    assert passages[0].startswith("[00:00–00:03]")
    assert "On screen: Pricing · Starter: $9 per month" in passages[0]
    assert "Speech: The starter plan is nine dollars." in passages[0]
    assert "Speech: We answer within four hours." in passages[1]


@needs_cv2
def test_speech_failure_keeps_the_on_screen_part(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cognivore.media.video import analyze_video

    if not (_FONT_DIR / "DejaVuSans.ttf").exists():
        pytest.skip("DejaVu font not installed")
    path = _write(tmp_path / "v.mp4", _hold(_slide("Hello"), 1))

    def broken(video_path: str, model_size: str, hint: str | None) -> Transcript:
        raise RuntimeError("no model download possible")

    monkeypatch.setattr("cognivore.media.video._has_audio_track", lambda _: True)
    analysis = analyze_video(path, transcriber=broken)

    assert analysis.transcript is None
    assert "speech_failed" in analysis.notes
    assert len(analysis.scenes) == 1


@needs_cv2
def test_a_video_without_sound_is_not_sent_to_speech_recognition(tmp_path: Path) -> None:
    from cognivore.media.video import analyze_video

    pytest.importorskip("av", reason="PyAV (installed with the audio extra) not installed")
    path = _write(tmp_path / "v.mp4", _hold(_slide("Hello"), 1))
    calls: list[str] = []

    analysis = analyze_video(path, transcriber=lambda p, m, h: calls.append(p))

    assert calls == []
    assert analysis.notes == []


def _analyze_with(
    monkeypatch: pytest.MonkeyPatch,
    scenes: list[Keyframe],
    transcript: Transcript | None,
    **kwargs: Any,
) -> tuple[VideoAnalysis, list[str | None]]:
    """analyze_video with scene detection and speech recognition faked, to
    test how the two are combined."""
    from cognivore.media import video

    monkeypatch.setattr(video, "extract_keyframes", lambda path, **_: scenes)
    monkeypatch.setattr(video, "_has_audio_track", lambda _: True)
    hints: list[str | None] = []

    def transcriber(path: str, model_size: str, hint: str | None) -> Transcript | None:
        hints.append(hint)
        return transcript

    return video.analyze_video("v.mp4", transcriber=transcriber, **kwargs), hints


def test_the_on_screen_language_tells_whisper_what_to_listen_for(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Measured: robotic Russian speech was guessed as Georgian (p=0.31) and
    came out as gibberish; with the language given, it came out Russian."""
    scenes = [Keyframe(0.0, 0, "Политика возврата\nПолный возврат в течение 14 дней", end=5.0)]
    _, hints = _analyze_with(monkeypatch, scenes, None, language_hint="en")

    assert hints == ["ru"]  # the slides win over the UI language


def test_without_on_screen_text_the_ui_language_is_the_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, hints = _analyze_with(monkeypatch, [Keyframe(0.0, 0, "", end=5.0)], None, language_hint="de")

    assert hints == ["de"]


def test_a_sound_track_without_speech_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    empty = Transcript(text="", segments=[], language="")
    analysis, _ = _analyze_with(monkeypatch, [Keyframe(0.0, 0, "Title", end=5.0)], empty)

    assert analysis.transcript is None
    assert analysis.notes == ["no_speech"]


def test_english_slides_get_english_labels_even_with_an_italian_looking_word() -> None:
    analysis = VideoAnalysis(
        duration=4, scenes=[Keyframe(0.0, 0, "Pricing\nStarter: $9 per month", end=4.0)]
    )
    assert timeline_passages(analysis)[0].splitlines()[1].startswith("On screen: ")


# -- Timeline (no OpenCV needed) ----------------------------------------


def _segment(start: float, end: float, text: str) -> TranscriptSegment:
    return TranscriptSegment(start=start, end=end, text=text)


def test_timeline_puts_speech_under_the_slide_shown_at_the_time() -> None:
    analysis = VideoAnalysis(
        duration=20.0,
        scenes=[
            Keyframe(0.0, 0, "Тарифы\nСтарт — 490 рублей", end=10.0),
            Keyframe(10.0, 100, "Безопасность", end=20.0),
        ],
        transcript=Transcript(
            text="",
            segments=[
                _segment(1, 4, "Тариф Старт стоит 490 рублей."),
                _segment(12, 15, "Всё шифруется."),
            ],
            language="ru",
        ),
    )
    sections = build_timeline(analysis)

    assert [(s.start, s.end) for s in sections] == [(0.0, 10.0), (10.0, 20.0)]
    assert sections[0].speech == "Тариф Старт стоит 490 рублей."
    assert sections[1].speech == "Всё шифруется."
    # Labels follow the language of the content.
    assert section_text(sections[0], "ru") == (
        "[00:00–00:10]\nНа экране: Тарифы · Старт — 490 рублей\nРечь: Тариф Старт стоит 490 рублей."
    )


def test_speech_without_slides_is_cut_into_minute_sections() -> None:
    segments = [_segment(t, t + 20, f"part {t}") for t in range(0, 150, 30)]
    analysis = VideoAnalysis(
        duration=150.0,
        scenes=[Keyframe(0.0, 0, "", end=150.0)],  # a talking head: no text
        transcript=Transcript(text="", segments=segments, language="en"),
    )
    sections = build_timeline(analysis)

    assert [s.on_screen for s in sections] == ["", "", ""]
    assert [round(s.start) for s in sections] == [0, 60, 120]
    assert "part 0 part 30" in sections[0].speech


def test_long_videos_get_hour_timestamps() -> None:
    analysis = VideoAnalysis(duration=4000, scenes=[Keyframe(3700.0, 0, "Q&A", end=4000.0)])
    assert timeline_passages(analysis)[0].startswith("[1:01:40–1:06:40]")


def test_summary_when_nothing_was_found() -> None:
    assert summarize_analysis(VideoAnalysis(duration=0, scenes=[])) == (
        "No distinct scenes were detected."
    )
    blank = VideoAnalysis(duration=5, scenes=[Keyframe(0.0, 0, "", end=5.0)])
    assert summarize_analysis(blank) == "No on-screen text or speech was found in this video."
