from __future__ import annotations

from typing import Any, ClassVar

from cognivore.media.audio import diarize_by_energy, transcribe
from cognivore.tools.base import Tool


class AudioTranscribeTool(Tool):
    name = "transcribe_audio"
    description = (
        "Transcribes speech in a local audio file (wav/mp3/m4a/...) to text, with rough "
        "speaker-turn labels. Give it a filesystem path."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {"audio_path": {"type": "string"}},
        "required": ["audio_path"],
    }

    def __init__(self, model_size: str = "base") -> None:
        self.model_size = model_size

    def run(self, audio_path: str = "", **_: object) -> str:
        try:
            transcript = transcribe(audio_path, model_size=self.model_size)
        except (RuntimeError, ValueError, OSError) as exc:
            return f"Error transcribing audio: {exc}"
        labeled = diarize_by_energy(transcript.segments)
        lines = [f"[{s.speaker}] {s.text}" for s in labeled]
        return f"Language: {transcript.language}\n" + "\n".join(lines)
