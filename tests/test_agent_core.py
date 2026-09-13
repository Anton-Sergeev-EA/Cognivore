from __future__ import annotations

from cognivore.agent.core import Agent
from cognivore.agent.parsing import parse_step
from cognivore.llm.base import GenerationConfig
from cognivore.llm.fake_backend import FakeLLMBackend
from cognivore.rag.embeddings import HashingEmbedder
from cognivore.rag.store import DocumentStore
from cognivore.tools.base import ToolRegistry
from cognivore.tools.calculator import CalculatorTool
from cognivore.tools.rag_search import RagSearchTool


def test_parse_step_final_answer() -> None:
    step = parse_step("Thought: easy one\nFinal Answer: 42")
    assert step.is_final
    assert step.final_answer == "42"


def test_parse_step_action() -> None:
    step = parse_step('Thought: need math\nAction: calculator\nAction Input: {"expression": "1+1"}')
    assert not step.is_final
    assert step.action == "calculator"
    assert step.action_input == {"expression": "1+1"}


def test_parse_step_falls_back_to_final_answer_on_malformed_output() -> None:
    step = parse_step("I don't know the format, just answering directly.")
    assert step.is_final
    assert "answering directly" in (step.final_answer or "")


def _build_agent(max_steps: int = 6) -> Agent:
    tools = ToolRegistry([CalculatorTool()])
    return Agent(llm=FakeLLMBackend(), tools=tools, max_steps=max_steps)


def test_agent_run_uses_calculator_tool_and_returns_result() -> None:
    agent = _build_agent()
    result = agent.run("What is 6 * 7?")
    assert result.answer.strip() == "42"
    assert any(t.action == "calculator" for t in result.trace)


def test_agent_run_no_tool_needed_returns_final_answer_directly() -> None:
    agent = _build_agent()
    result = agent.run("Just say hello.")
    assert not result.hit_step_limit
    assert result.trace  # at least the final thought entry
    assert result.trace[-1].action is None


def test_agent_remembers_conversation_turns() -> None:
    agent = _build_agent()
    agent.run("2 + 2")
    assert len(agent.conversation.as_list()) == 2  # one user turn + one assistant turn


def test_agent_dispatch_unknown_tool_reports_error_without_crashing() -> None:
    tools = ToolRegistry([CalculatorTool()])
    agent = Agent(llm=FakeLLMBackend(), tools=tools)
    from cognivore.agent.parsing import AgentStep

    fake_step = AgentStep(thought="x", action="not_a_real_tool", action_input={}, final_answer=None)
    observation = agent._dispatch(fake_step)
    assert "unknown tool" in observation.lower()


def test_agent_with_rag_search_tool_answers_from_ingested_document() -> None:
    store = DocumentStore(embedder=HashingEmbedder(dim=64))
    store.add_text(
        "The Eiffel Tower is located in Paris, France, and was completed in 1889.",
        source="facts.md",
    )
    tools = ToolRegistry([RagSearchTool(store)])
    agent = Agent(llm=FakeLLMBackend(), tools=tools, generation_config=GenerationConfig())
    result = agent.run("search the knowledge base for where the Eiffel Tower is")
    assert "Paris" in result.answer
    assert any(t.action == "search_knowledge_base" for t in result.trace)


def test_run_stream_yields_answer_deltas_that_reassemble_to_the_final_answer() -> None:
    agent = _build_agent()
    events = list(agent.run_stream("Just say hello."))

    deltas = "".join(payload for kind, payload in events if kind == "answer_delta")
    finals = [payload for kind, payload in events if kind == "final"]
    assert len(finals) == 1
    assert deltas.strip() == finals[0].answer.strip()


def test_run_stream_emits_trace_before_streaming_the_final_answer() -> None:
    agent = _build_agent()
    events = list(agent.run_stream("What is 6 * 7?"))

    kinds = [kind for kind, _ in events]
    assert "trace" in kinds
    assert kinds.index("trace") < kinds.index("answer_delta")
    finals = [payload for kind, payload in events if kind == "final"]
    assert finals[0].answer.strip() == "42"


def test_agent_falls_back_to_users_own_wording_when_model_translates_the_query() -> None:
    # Simulate what was actually observed with a real local model: it
    # rewrites/translates the search query it gives the tool rather than
    # reusing the user's own words. The RAG search tool should still find
    # the answer by also trying the original, untranslated question.
    store = DocumentStore(embedder=HashingEmbedder(dim=64))
    store.add_text(
        "Возврат средств возможен в течение 14 дней с момента оплаты.",  # noqa: RUF001
        source="policy.md",
    )
    tools = ToolRegistry([RagSearchTool(store)])

    class _TranslatingFakeLLM:
        def generate(self, messages, config):
            last_user = messages[-1].content
            if last_user.startswith("Observation:"):
                return f"Thought: done.\nFinal Answer: {last_user[len('Observation:') :].strip()}"
            return (
                "Thought: translating for the search.\nAction: search_knowledge_base\n"
                'Action Input: {"query": "refund policy within a week of payment"}'
            )

        def stream(self, messages, config):
            yield self.generate(messages, config)

    agent = Agent(llm=_TranslatingFakeLLM(), tools=tools)
    result = agent.run("Какой возврат средств в первую неделю после оплаты?")
    assert "14 дней" in result.answer


def test_run_stream_does_not_leak_raw_react_syntax_into_answer_deltas() -> None:
    # A tool-call step's raw completion (e.g. 'Action: calculator\n...')
    # must never appear in an answer_delta -- only text from a genuine
    # "Final Answer:" section should stream live.
    agent = _build_agent()
    events = list(agent.run_stream("What is 6 * 7?"))

    deltas = "".join(payload for kind, payload in events if kind == "answer_delta")
    assert "Action:" not in deltas
    assert "Thought:" not in deltas
