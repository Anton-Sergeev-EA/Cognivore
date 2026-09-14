"""Correctness tests for lightweight video understanding.

Both dependencies are optional (``cognivore[video]``), and OCR additionally
needs the native Tesseract *binary* on top of the ``pytesseract`` Python
package -- so every test here self-skips when either piece isn't available,
rather than failing, the same pattern ``test_index.py`` uses for the native
extension. That keeps the suite green on a machine that only installed the
base package, while still giving real, non-mocked coverage wherever the
video extra (and Tesseract) actually are installed -- e.g. the Docker image,
which now installs both.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

cv2 = pytest.importorskip(
    "cv2", reason="opencv-python-headless not installed (pip install cognivore[video])"
)

pytest.importorskip(
    "pytesseract", reason="pytesseract not installed (pip install cognivore[video])"
)

from cognivore.media.video import _ocr_frame, extract_keyframes, summarize_keyframes  # noqa: E402

tesseract_missing = shutil.which("tesseract") is None
pytestmark_ocr = pytest.mark.skipif(
    tesseract_missing,
    reason="tesseract binary not installed (apt install tesseract-ocr / brew install tesseract)",
)


def _make_two_scene_video(path: Path) -> None:
    """A tiny synthetic .mp4: 1s of plain gray, then 1s of a dark frame with
    burned-in white text -- enough to exercise scene-change detection and,
    on the second scene, OCR, without needing a real video fixture checked
    into the repo."""
    w, h, fps = 320, 240, 10
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    try:
        import numpy as np

        for _ in range(fps):
            writer.write(np.full((h, w, 3), 200, dtype=np.uint8))
        for _ in range(fps):
            frame = np.full((h, w, 3), 30, dtype=np.uint8)
            cv2.putText(
                frame, "SCENE TWO", (20, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3
            )
            writer.write(frame)
    finally:
        writer.release()


def test_extract_keyframes_detects_scene_changes(tmp_path: Path) -> None:
    video_path = tmp_path / "clip.mp4"
    _make_two_scene_video(video_path)

    # A low threshold because the synthetic frames above are flat colors --
    # their histograms are near-delta functions, so even a drastic-looking
    # cut produces a smaller chi-square distance than typical real footage
    # would. Real video content should generally work fine with the
    # documented default of 30.0.
    keyframes = extract_keyframes(str(video_path), scene_threshold=0.5, run_ocr=False)

    assert len(keyframes) == 2
    assert keyframes[0].frame_index == 0
    assert keyframes[1].frame_index == 10


def test_extract_keyframes_respects_max_keyframes(tmp_path: Path) -> None:
    video_path = tmp_path / "clip.mp4"
    _make_two_scene_video(video_path)

    keyframes = extract_keyframes(
        str(video_path), scene_threshold=0.5, max_keyframes=1, run_ocr=False
    )

    assert len(keyframes) == 1


def test_extract_keyframes_raises_on_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        extract_keyframes(str(tmp_path / "does-not-exist.mp4"))


@pytestmark_ocr
def test_ocr_frame_reads_text_off_a_synthetic_frame() -> None:
    import numpy as np

    frame = np.full((100, 400, 3), 255, dtype=np.uint8)
    cv2.putText(frame, "HELLO WORLD", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)

    assert "HELLO" in _ocr_frame(frame).upper()


@pytestmark_ocr
def test_extract_keyframes_runs_ocr_on_the_second_scene(tmp_path: Path) -> None:
    video_path = tmp_path / "clip.mp4"
    _make_two_scene_video(video_path)

    keyframes = extract_keyframes(str(video_path), scene_threshold=0.5, run_ocr=True)

    assert len(keyframes) == 2
    assert keyframes[0].ocr_text == ""
    assert "SCENE TWO" in keyframes[1].ocr_text.upper()


def test_ocr_frame_returns_empty_string_when_pytesseract_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`_ocr_frame` must degrade to "" rather than raise when the optional
    OCR dependency isn't installed -- this is what made the missing
    `pytesseract` dependency (before this fix) fail silently instead of
    loudly, so it's worth locking in as the *intended* fallback behavior for
    a genuinely-missing dependency, as opposed to a declared-but-uninstalled
    one."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "pytesseract":
            raise ImportError("simulated: pytesseract not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    import numpy as np

    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    assert _ocr_frame(frame) == ""


def test_summarize_keyframes_includes_ocr_text_when_present() -> None:
    from cognivore.media.video import Keyframe

    summary = summarize_keyframes([Keyframe(timestamp=1.0, frame_index=10, ocr_text="SCENE TWO")])

    assert "00:01" in summary
    assert "SCENE TWO" in summary


def test_summarize_keyframes_handles_no_scenes() -> None:
    assert summarize_keyframes([]) == "No distinct scenes were detected."
