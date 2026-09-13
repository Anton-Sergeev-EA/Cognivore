"""Parses a model's ReAct-formatted completion into a structured step.

Kept deliberately forgiving: local, small, CPU-friendly models don't always
follow formatting instructions perfectly, so this accepts minor variations
(extra whitespace, a trailing code fence around the JSON, "action input"
without exact casing) rather than hard-failing on the first mismatch --
falling back to treating the whole completion as a final answer is much
better UX than raising deep inside an agent loop.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

_THOUGHT_RE = re.compile(r"Thought:\s*(.*?)(?=\n(?:Action|Final Answer):|\Z)", re.S | re.I)
_ACTION_RE = re.compile(r"Action:\s*([^\n]+)", re.I)
_ACTION_INPUT_RE = re.compile(
    r"Action Input:\s*(\{.*?\})\s*(?=\n(?:Thought|Observation):|\Z)", re.S | re.I
)
_FINAL_ANSWER_RE = re.compile(r"Final Answer:\s*(.*)", re.S | re.I)


@dataclass
class AgentStep:
    thought: str
    action: str | None
    action_input: dict
    final_answer: str | None

    @property
    def is_final(self) -> bool:
        return self.final_answer is not None


def parse_step(completion: str) -> AgentStep:
    thought_match = _THOUGHT_RE.search(completion)
    thought = thought_match.group(1).strip() if thought_match else ""

    final_match = _FINAL_ANSWER_RE.search(completion)
    action_match = _ACTION_RE.search(completion)

    if final_match and (not action_match or final_match.start() < action_match.start()):
        return AgentStep(
            thought=thought, action=None, action_input={}, final_answer=final_match.group(1).strip()
        )

    if action_match:
        action = action_match.group(1).strip().strip("`")
        input_match = _ACTION_INPUT_RE.search(completion)
        action_input = _parse_json_loose(input_match.group(1)) if input_match else {}
        return AgentStep(
            thought=thought, action=action, action_input=action_input, final_answer=None
        )

    # Nothing matched the expected format: treat the whole thing as the
    # final answer rather than erroring out.
    return AgentStep(thought=thought, action=None, action_input={}, final_answer=completion.strip())


def _parse_json_loose(raw: str) -> dict:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.M).strip()
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}
