from __future__ import annotations

from typing import Any, ClassVar

from cognivore.media.video import extract_keyframes, summarize_keyframes
from cognivore.tools.base import Tool


class VideoAnalyzeTool(Tool):
    name = "analyze_video"
    description = (
        "Analyzes a local video file: detects scene changes and extracts any on-screen text "
        "(OCR) at each detected scene, with timestamps. Give it a filesystem path. Good for "
        "screencasts, lecture recordings, and slide-based videos; combine with "
        "transcribe_audio for the spoken narration."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {"video_path": {"type": "string"}},
        "required": ["video_path"],
    }

    def __init__(
        self,
        scene_threshold: float = 30.0,
        max_keyframes: int = 24,
        ocr_languages: str = "eng+rus",
    ) -> None:
        self.scene_threshold = scene_threshold
        self.max_keyframes = max_keyframes
        self.ocr_languages = ocr_languages

    def run(self, video_path: str = "", **_: object) -> str:
        try:
            keyframes = extract_keyframes(
                video_path,
                scene_threshold=self.scene_threshold,
                max_keyframes=self.max_keyframes,
                ocr_languages=self.ocr_languages,
            )
        except (RuntimeError, ValueError, OSError) as exc:
            return f"Error analyzing video: {exc}"
        return summarize_keyframes(keyframes)
