"""Lightweight, CPU-only video understanding.

Rather than reaching for a heavyweight vision-language model (which
typically means dragging in PyTorch and a multi-GB checkpoint -- a poor fit
for "runs comfortably on a CPU-only laptop"), this module extracts the
signal that's cheap to get and often enough to *understand* a video without
truly "seeing" it:

  * keyframes via histogram-based scene-change detection (OpenCV),
  * on-screen text via OCR on those keyframes (if ``pytesseract`` + the
    Tesseract binary are available),
  * and, combined with :mod:`cognivore.media.audio`'s transcript, the LLM
    is handed timestamps + OCR text + speech -- which for the common cases
    (lectures, tutorials, meeting recordings, screencasts) captures most of
    what a caption model would tell you anyway, at a fraction of the compute.

Swapping in a real captioning model later is a matter of adding a
``caption_frame(frame) -> str`` function and calling it inside
``extract_keyframes``; the rest of the pipeline doesn't need to change.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Keyframe:
    timestamp: float
    frame_index: int
    ocr_text: str = ""


def _require_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "opencv-python-headless is not installed. Install it with `pip install cognivore[video]`."
        ) from exc
    return cv2


def extract_keyframes(
    video_path: str,
    scene_threshold: float = 30.0,
    max_keyframes: int = 24,
    run_ocr: bool = True,
    ocr_languages: str = "eng+rus",
) -> list[Keyframe]:
    cv2 = _require_cv2()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    prev_hist = None
    keyframes: list[Keyframe] = []
    frame_index = 0

    try:
        while cap.isOpened() and len(keyframes) < max_keyframes:
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
            cv2.normalize(hist, hist)

            is_scene_change = prev_hist is None
            if prev_hist is not None:
                diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CHISQR)
                is_scene_change = diff > scene_threshold

            if is_scene_change:
                text = _ocr_frame(frame, lang=ocr_languages) if run_ocr else ""
                keyframes.append(
                    Keyframe(timestamp=frame_index / fps, frame_index=frame_index, ocr_text=text)
                )
                prev_hist = hist

            frame_index += 1
    finally:
        cap.release()

    return keyframes


def _ocr_frame(frame, lang: str = "eng+rus") -> str:
    try:
        import pytesseract
    except ImportError:
        return ""
    try:
        return pytesseract.image_to_string(frame, lang=lang).strip()
    except Exception:  # pragma: no cover - depends on the tesseract binary being installed
        return ""


def summarize_keyframes(keyframes: list[Keyframe]) -> str:
    if not keyframes:
        return "No distinct scenes were detected."
    lines = []
    for kf in keyframes:
        mm, ss = divmod(int(kf.timestamp), 60)
        line = f"[{mm:02d}:{ss:02d}] scene change"
        if kf.ocr_text:
            line += f" -- on-screen text: {kf.ocr_text[:200]!r}"
        lines.append(line)
    return "\n".join(lines)
