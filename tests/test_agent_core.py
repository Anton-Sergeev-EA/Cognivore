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
        "Возврат средств возможен в течение 14 дней с момента оплаты.",
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


def _rag_agent() -> Agent:
    store = DocumentStore(embedder=HashingEmbedder(dim=64))
    store.add_text(
        "## Политика возврата\n**Полный** возврат в течение 14 дней с момента оплаты.",
        source="Demo KB (RU)",
    )
    tools = ToolRegistry([CalculatorTool(), RagSearchTool(store)])
    return Agent(llm=FakeLLMBackend(), tools=tools)


def test_offline_demo_searches_for_plain_questions_in_any_language() -> None:
    agent = _rag_agent()
    for question in ("Какая политика возврата?", "退款政策是什么？", "What is the refund policy?"):
        result = agent.run(question)
        assert any(t.action == "search_knowledge_base" for t in result.trace), question


def test_offline_demo_answers_with_the_top_passage_only() -> None:
    # Source labels contain parentheses ("Demo KB (RU)"), and passages may
    # contain markdown; neither may leak into the answer.
    result = _rag_agent().run("Какая политика возврата?")
    assert result.answer.startswith("Политика возврата")
    assert "[1]" not in result.answer
    assert "**" not in result.answer and "##" not in result.answer


def test_offline_demo_understands_percentages() -> None:
    agent = _build_agent()
    assert agent.run("Сколько будет 15% от 4900?").answer.strip() == "735"
    assert agent.run("What is 15% of 149?").answer.strip() == "22.35"
    assert agent.run("999 的 15% 是多少？").answer.strip() == "149.85"
    assert agent.run("Wie viel sind 15 % von 149?").answer.strip() == "22.35"
    assert agent.run("¿Cuánto es el 15 % de 149?").answer.strip() == "22.35"
    assert agent.run("Combien font 15 % de 149 ?").answer.strip() == "22.35"
    assert agent.run("Quanto fa il 15% di 149?").answer.strip() == "22.35"
    assert agent.run("149 の 15% はいくつ？").answer.strip() == "22.35"
    assert agent.run("149 का 15% कितना है?").answer.strip() == "22.35"


def test_offline_demo_does_not_call_search_when_it_is_not_registered() -> None:
    result = _build_agent().run("What is the capital of France?")
    assert all(t.action is None for t in result.trace)


class _RecordingLLM:
    """Answers immediately (never calls a tool) and remembers its prompt --
    the behaviour observed live from a 3B model."""

    def __init__(self, answer: str = "Final Answer: ok") -> None:
        self.answer = answer
        self.messages: list = []

    def generate(self, messages, config):
        self.messages = list(messages)
        return self.answer

    def stream(self, messages, config):
        yield self.generate(messages, config)


def _system_text(llm: _RecordingLLM) -> str:
    return "\n".join(m.content for m in llm.messages if m.role == "system")


def test_context_provider_puts_passages_in_front_of_the_model() -> None:
    llm = _RecordingLLM()
    agent = Agent(
        llm=llm,
        tools=ToolRegistry([CalculatorTool()]),
        context_provider=lambda q: "[1] (policy.md) Full refund within 14 days.",
    )
    result = agent.run("What is the refund policy?")
    assert "Full refund within 14 days." in _system_text(llm)
    # Recorded as a normal search step, so the UI and insight layer see it.
    assert result.trace[0].action == "search_knowledge_base"
    assert result.trace[0].observation == "[1] (policy.md) Full refund within 14 days."


def test_context_provider_returning_none_adds_nothing() -> None:
    llm = _RecordingLLM()
    agent = Agent(llm=llm, tools=ToolRegistry([CalculatorTool()]), context_provider=lambda q: None)
    result = agent.run("What is 6 * 7?")
    assert "Passages retrieved" not in _system_text(llm)
    assert all(t.action is None for t in result.trace)


def test_run_stream_emits_the_auto_search_step_first() -> None:
    agent = Agent(
        llm=_RecordingLLM(),
        tools=ToolRegistry([CalculatorTool()]),
        context_provider=lambda q: "[1] (a.md) text",
    )
    events = list(agent.run_stream("What is the refund policy?"))
    assert events[0][0] == "trace"
    assert events[0][1].action == "search_knowledge_base"


def test_answer_language_is_named_explicitly() -> None:
    for question, name in [
        ("Какая политика возврата средств?", "Russian"),
        ("¿Cuál es la política de reembolso?", "Spanish"),
        ("Wie lautet die Rückerstattungsrichtlinie?", "German"),
        ("退款政策是什么？", "Chinese"),
    ]:
        llm = _RecordingLLM()
        Agent(llm=llm, tools=ToolRegistry([CalculatorTool()])).run(question)
        assert f"entirely in {name}" in _system_text(llm), question


def test_no_language_instruction_without_a_clear_signal() -> None:
    llm = _RecordingLLM()
    Agent(llm=llm, tools=ToolRegistry([CalculatorTool()])).run("12 * 7")
    assert "entirely in" not in _system_text(llm)


def test_empty_lookup_tells_the_model_the_knowledge_base_has_nothing() -> None:
    llm = _RecordingLLM()
    agent = Agent(llm=llm, tools=ToolRegistry([CalculatorTool()]), context_provider=lambda q: "")
    result = agent.run("Do you have a mobile app for iPhone?")
    assert "contains nothing about it" in _system_text(llm)
    assert "do not tell them to search elsewhere" in _system_text(llm)
    assert result.trace[0].action == "search_knowledge_base"
    assert result.trace[0].observation == "No relevant passages found."


def test_search_tool_reports_nothing_found_instead_of_unrelated_passages() -> None:
    store = DocumentStore(embedder=HashingEmbedder(dim=64))
    store.add_text("A full refund is available within 14 days of the first payment.", "p.md")
    strict = RagSearchTool(store, min_confidence=0.3)
    assert strict.run(query="mobile app for iPhone") == "No relevant passages found."
    assert "14 days" in strict.run(query="refund policy")
    # Without a threshold (the default) the tool keeps returning its best hits.
    assert "14 days" in RagSearchTool(store).run(query="mobile app for iPhone")


def test_final_answer_written_as_a_tool_call_is_still_the_final_answer() -> None:
    # Live finding with qwen2.5:3b: "Action: Final Answer" used to be
    # dispatched as a tool, and the model relayed the resulting
    # "unknown tool" error to the user as its answer.
    cases = [
        (
            'Thought: none\nAction: Final Answer\nAction Input: {"answer": "Нет информации."}',
            "Нет информации.",
        ),
        (
            "Action: final_answer\nAction Input: The knowledge base has nothing on this.",
            "The knowledge base has nothing on this.",
        ),
        ('Action: Final Answer\nAction Input: "Plain JSON string."', "Plain JSON string."),
        (
            'Action: `Final Answer:`\nAction Input: {"response": "Via another key."}',
            "Via another key.",
        ),
    ]
    for completion, expected in cases:
        step = parse_step(completion)
        assert step.is_final, completion
        assert step.final_answer == expected


def test_agent_does_not_relay_an_unknown_tool_error_for_a_final_answer_call() -> None:
    class _ConfusedLLM:
        def generate(self, messages, config):
            return (
                "Thought: The knowledge base has nothing.\nAction: Final Answer\n"
                'Action Input: {"answer": "В базе знаний нет информации об этом."}'
            )

        def stream(self, messages, config):
            yield self.generate(messages, config)

    result = Agent(llm=_ConfusedLLM(), tools=ToolRegistry([CalculatorTool()])).run("Вопрос?")
    assert result.answer == "В базе знаний нет информации об этом."
    assert all(t.action is None for t in result.trace)


def test_strict_mode_answers_an_empty_lookup_without_the_model() -> None:
    llm = _RecordingLLM()
    agent = Agent(
        llm=llm,
        tools=ToolRegistry([CalculatorTool()]),
        context_provider=lambda q: "",
        strict_knowledge_answers=True,
    )
    result = agent.run("Есть ли у вас мобильное приложение?")
    assert result.answer == "В базе знаний нет информации об этом."
    assert llm.messages == []  # the model was never called
    assert result.trace[0].action == "search_knowledge_base"

    events = list(agent.run_stream("Gibt es eine Handy-App?"))
    deltas = "".join(p for kind, p in events if kind == "answer_delta")
    assert deltas == "Die Wissensbasis enthält dazu keine Informationen."
    assert events[-1][0] == "final"


def test_strict_mode_still_uses_the_model_when_passages_were_found() -> None:
    llm = _RecordingLLM("Final Answer: 14 days.")
    agent = Agent(
        llm=llm,
        tools=ToolRegistry([CalculatorTool()]),
        context_provider=lambda q: "[1] (p.md) Full refund within 14 days.",
        strict_knowledge_answers=True,
    )
    assert agent.run("What is the refund policy?").answer == "14 days."
    assert llm.messages
