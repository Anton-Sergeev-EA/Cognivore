from __future__ import annotations

from typing import Any, ClassVar

from cognivore.media.video import analyze_video, summarize_analysis
from cognivore.tools.base import Tool


class VideoAnalyzeTool(Tool):
    name = "analyze_video"
    description = (
        "Analyzes a local video file: finds its scenes (slides, screens), reads the on-screen "
        "text of each and transcribes the speech, as a timeline with timestamps. Give it a "
        "filesystem path. Good for lectures, screencasts, meetings and slide-based videos."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {"video_path": {"type": "string"}},
        "required": ["video_path"],
    }

    def __init__(
        self,
        scene_threshold: float = 30.0,
        max_keyframes: int = 200,
        ocr_languages: str = "auto",
        whisper_model_size: str = "small",
    ) -> None:
        self.scene_threshold = scene_threshold
        self.max_keyframes = max_keyframes
        self.ocr_languages = ocr_languages
        self.whisper_model_size = whisper_model_size

    def run(self, video_path: str = "", **_: object) -> str:
        try:
            analysis = analyze_video(
                video_path,
                scene_threshold=self.scene_threshold,
                max_keyframes=self.max_keyframes,
                ocr_languages=self.ocr_languages,
                whisper_model_size=self.whisper_model_size,
            )
        except (RuntimeError, ValueError, OSError) as exc:
            return f"Error analyzing video: {exc}"
        return summarize_analysis(analysis)
