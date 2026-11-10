from pathlib import Path

import pytest

from cognivore.agent.core import Agent
from cognivore.agent.runs import RunStore
from cognivore.llm.base import ChatMessage, GenerationConfig
from cognivore.llm.fake_backend import FakeLLMBackend
from cognivore.tools.base import ToolRegistry
from cognivore.tools.calculator import CalculatorTool


class InterruptAfterObservation(FakeLLMBackend):
    def generate(self, messages: list[ChatMessage], config: GenerationConfig) -> str:
        if any(message.content.startswith("Observation:") for message in messages):
            raise InterruptedError("Stopped after a completed tool call.")
        return super().generate(messages, config)


class NoReplayCalculator(CalculatorTool):
    def run(self, expression: str = "", **_: object) -> str:
        pytest.fail("A completed tool call was replayed.")


def test_resume_after_tool_call_and_publish_once(tmp_path: Path) -> None:
    path = tmp_path / "runs.sqlite3"
    runs = RunStore(path)
    run_id = runs.create_task("What is 6 * 7?")
    agent = Agent(InterruptAfterObservation(), ToolRegistry([CalculatorTool()]))
    with pytest.raises(InterruptedError):
        runs.resume_task(run_id, agent)

    reopened = RunStore(path)
    agent = Agent(FakeLLMBackend(), ToolRegistry([NoReplayCalculator()]))
    result = reopened.resume_task(run_id, agent)
    assert result.report == "42"
    assert result.status == "pending"
    assert not (tmp_path / "reports").exists()

    published = reopened.approve(run_id)
    assert published.read_text(encoding="utf-8") == "42"
    assert RunStore(path).approve(run_id) == published
    assert reopened.resume_task(run_id, agent).status == "approved"
