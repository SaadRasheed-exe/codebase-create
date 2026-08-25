import json
from types import SimpleNamespace as NS

import pytest

from codebase_create.config import AgentConfig
from codebase_create.executor import TempWorkspace
from codebase_create.models import (
    AssistantMessage,
    ToolCall,
    ToolResult,
    ToolResultMessage,
    ToolSpec,
    UserMessage,
)
from codebase_create.providers import BACKENDS, build_provider
from codebase_create.providers.anthropic_provider import (
    anthropic_tools_from_specs,
    parse_anthropic_response,
    to_anthropic_messages,
)
from codebase_create.providers.base import ProviderError
from codebase_create.providers.mock import MockProvider
from codebase_create.providers.openai_compat import (
    openai_tools_from_specs,
    parse_openai_response,
    to_openai_messages,
)
from codebase_create.tools import TOOL_SPECS, ToolDispatcher


# ---------------------------------------------------------------------------
# helpers: a minimal loop driving mock scenarios through the real dispatcher
# ---------------------------------------------------------------------------


def play_scenario(scenario: str, max_turns: int = 20):
    """Drives a scripted scenario through the real dispatcher.

    Returns (final_reply_or_None, all_observations). final_reply is None
    when the script exhausted without a text-only finish (e.g.
    stuck_loop, whose termination is the orchestrator detector's job).
    """
    ws = TempWorkspace()
    try:
        dispatcher = ToolDispatcher(ws, AgentConfig(sandbox="subprocess"))
        provider = MockProvider(scenario)
        history: list = []
        observations: list[ToolResult] = []
        for _ in range(max_turns):
            try:
                reply = provider.complete("system", history, TOOL_SPECS)
            except ProviderError:
                return None, observations
            if not reply.tool_calls:
                return reply, observations
            results = [dispatcher.execute(call) for call in reply.tool_calls]
            observations.extend(results)
            history.append(reply)
            history.append(ToolResultMessage(results=results))
        raise AssertionError(f"scenario '{scenario}' did not finish")
    finally:
        ws.cleanup()


# ---------------------------------------------------------------------------
# mock scenarios end-to-end (offline, deterministic)
# ---------------------------------------------------------------------------


def test_happy_path_passes_cleanly():
    reply, observations = play_scenario("happy_path")
    assert not reply.tool_calls
    assert observations[-1].content == "PASSED: 3 test(s) passed."
    assert all(not obs.is_error for obs in observations)


def test_fix_after_failure_repairs_and_passes():
    reply, observations = play_scenario("fix_after_failure")
    assert "PASSED" in observations[-1].content
    failed_runs = [o for o in observations if o.content.startswith("FAILED")]
    assert failed_runs, "scenario must include at least one failing run"
    read_calls = [o for o in observations if o.name == "read_file"]
    assert len(read_calls) == 1  # context-aware repair reads before patching


def test_stuck_loop_produces_identical_failures():
    _reply, observations = play_scenario("stuck_loop")
    run_results = [o for o in observations if o.name == "run_tests"]
    assert len(run_results) >= 3
    fingerprints = {r.content.splitlines()[0] for r in run_results[:3]}
    assert len(fingerprints) == 1  # identical verdicts feed the stuck detector


def test_bad_tool_args_survives_errors_and_recovers():
    _reply, observations = play_scenario("bad_tool_args")
    errors = [o for o in observations if o.is_error]
    assert len(errors) == 2
    assert any("argument 'content'" in o.content for o in errors)
    assert any("escapes the workspace" in o.content for o in errors)
    assert observations[-1].content == "PASSED: 3 test(s) passed."


def test_unknown_scenario_lists_valid_options():
    with pytest.raises(ProviderError, match="happy_path"):
        MockProvider("no_such_scenario")


def test_script_exhaustion_raises_loudly(monkeypatch):
    monkeypatch.setitem(
        __import__(
            "codebase_create.providers.mock_scenarios", fromlist=["SCENARIOS"]
        ).SCENARIOS,
        "tiny",
        [__import__("codebase_create.providers.mock_scenarios", fromlist=["ScriptedTurn"]).ScriptedTurn()],
    )
    provider = MockProvider("tiny")
    provider.complete("s", [], [])
    with pytest.raises(ProviderError, match="exhausted"):
        provider.complete("s", [], [])


def test_conformance_check_rejects_unknown_tool_in_scenario(monkeypatch):
    scenarios_mod = __import__(
        "codebase_create.providers.mock_scenarios", fromlist=["SCENARIOS", "ScriptedTurn"]
    )
    bogus = scenarios_mod.ScriptedTurn(tool_calls=[("nuke_everything", {})])
    monkeypatch.setitem(scenarios_mod.SCENARIOS, "bogus", [bogus])
    with pytest.raises(ProviderError, match="unknown tool 'nuke_everything'"):
        MockProvider("bogus")


def test_mock_ignores_history_and_temperature():
    p1, p2 = MockProvider(), MockProvider()
    junk_history = [UserMessage("completely different request")]
    a = p1.complete("sysA", junk_history, TOOL_SPECS, temperature=0.9)
    b = p2.complete("sysB", [], TOOL_SPECS, temperature=0.0)
    assert a == b


# ---------------------------------------------------------------------------
# OpenAI-compatible wire format conversions (pure, offline)
# ---------------------------------------------------------------------------


def sample_conversation():
    return [
        UserMessage("build factorial"),
        AssistantMessage(
            text="writing files",
            tool_calls=[ToolCall(id="c1", name="write_file",
                                 arguments={"path": "solution.py"})],
        ),
        ToolResultMessage(results=[
            ToolResult(call_id="c1", name="write_file", content="Wrote 5 bytes"),
            ToolResult(call_id="c2", name="run_tests", content="boom", is_error=True),
        ]),
        AssistantMessage(text="done"),
    ]


def test_openai_message_conversion_roundtrip():
    dicts = to_openai_messages("be terse", sample_conversation())

    assert dicts[0] == {"role": "system", "content": "be terse"}
    assert dicts[1]["role"] == "user"
    assistant = dicts[2]
    assert assistant["role"] == "assistant"
    assert assistant["tool_calls"][0]["function"]["name"] == "write_file"
    # arguments survive a JSON serialize/parse round trip
    parsed = json.loads(assistant["tool_calls"][0]["function"]["arguments"])
    assert parsed == {"path": "solution.py"}
    tool_msgs = [d for d in dicts if d["role"] == "tool"]
    assert tool_msgs[0]["content"] == "Wrote 5 bytes"
    assert tool_msgs[1]["content"].startswith("[error] ")
    assert tool_msgs[0]["tool_call_id"] == "c1"


def test_openai_assistant_without_text_uses_null_content():
    dicts = to_openai_messages("s", [
        AssistantMessage(text="", tool_calls=[ToolCall("i", "list_files", {})]),
    ])
    assert dicts[1]["content"] is None


def test_openai_tools_mapping_and_shape():
    specs = [ToolSpec(name="f", description="d", parameters={"type": "object"})]
    mapped = openai_tools_from_specs(specs)
    assert mapped[0]["function"]["parameters"] == {"type": "object"}
    assert openai_tools_from_specs([]) == []


def test_parse_openai_response_extracts_calls_and_usage():
    fake = NS(
        choices=[NS(message=NS(
            content=None,
            tool_calls=[NS(id="z9", function=NS(name="read_file",
                                                arguments='{"path": "a.py"}'))],
        ))],
        usage=NS(prompt_tokens=120, completion_tokens=34),
    )
    msg = parse_openai_response(fake)
    assert msg.tool_calls == [ToolCall(id="z9", name="read_file",
                                       arguments={"path": "a.py"})]
    assert (msg.input_tokens, msg.output_tokens) == (120, 34)


def test_parse_openai_rejects_malformed_arguments():
    fake = NS(choices=[NS(message=NS(content=None, tool_calls=[
        NS(id="q", function=NS(name="write_file", arguments="{not json")),
    ]))], usage=None)
    with pytest.raises(ProviderError, match="malformed JSON"):
        parse_openai_response(fake)


def test_parse_openai_captures_reasoning():
    fake = NS(
        choices=[NS(message=NS(
            content="answer",
            reasoning="step by step",
            tool_calls=None,
        ))],
        usage=NS(prompt_tokens=50, completion_tokens=40),
    )
    msg = parse_openai_response(fake)
    assert msg.thinking == "step by step"
    assert msg.text == "answer"
    assert msg.thinking_tokens == 0


def test_parse_openai_subtracts_reasoning_tokens():
    fake = NS(
        choices=[NS(message=NS(
            content="answer",
            reasoning="thinking",
            tool_calls=None,
        ))],
        usage=NS(
            prompt_tokens=100,
            completion_tokens=80,
            completion_tokens_details=NS(reasoning_tokens=50),
        ),
    )
    msg = parse_openai_response(fake)
    assert msg.thinking == "thinking"
    assert msg.output_tokens == 30  # 80 - 50
    assert msg.thinking_tokens == 50


# ---------------------------------------------------------------------------
# Anthropic wire format conversions (pure, offline)
# ---------------------------------------------------------------------------


def test_anthropic_message_conversion_batches_tool_results():
    dicts = to_anthropic_messages(sample_conversation())
    roles = [d["role"] for d in dicts]
    assert roles == ["user", "assistant", "user", "assistant"]

    tool_result_block = dicts[2]["content"][0]
    assert tool_result_block["type"] == "tool_result"
    assert tool_result_block["is_error"] is False

    error_blocks = [
        b for d in dicts for b in (d["content"] if isinstance(d["content"], list) else [])
        if b.get("type") == "tool_result" and b.get("is_error")
    ]
    assert len(error_blocks) == 1

    assistant_blocks = dicts[1]["content"]
    assert {"type": "text", "text": "writing files"} in assistant_blocks
    assert any(b.get("type") == "tool_use" and b["name"] == "write_file"
               for b in assistant_blocks)


def test_anthropic_tools_mapping_uses_input_schema():
    spec = ToolSpec(name="f", description="d", parameters={"type": "object"})
    mapped = anthropic_tools_from_specs([spec])
    assert mapped[0]["input_schema"] == {"type": "object"}
    assert "parameters" not in mapped[0]


def test_parse_anthropic_response_merges_text_and_calls():
    fake = NS(
        content=[
            NS(type="text", text="line one"),
            NS(type="text", text="line two"),
            NS(type="tool_use", id="tu_1", name="run_tests", input={}),
            NS(type="thinking", thinking="hidden"),
        ],
        usage=NS(input_tokens=10, output_tokens=20),
    )
    msg = parse_anthropic_response(fake)
    assert msg.text == "line one\nline two"
    assert msg.thinking == "hidden"
    assert msg.tool_calls == [ToolCall(id="tu_1", name="run_tests", arguments={})]
    assert (msg.input_tokens, msg.output_tokens) == (10, 20)
    assert msg.thinking_tokens == 0


def test_parse_anthropic_subtracts_thinking_tokens():
    fake = NS(
        content=[
            NS(type="thinking", thinking="reasoning"),
            NS(type="text", text="answer"),
        ],
        usage=NS(
            input_tokens=100,
            output_tokens=50,
            output_tokens_details=NS(thinking_tokens=30),
        ),
    )
    msg = parse_anthropic_response(fake)
    assert msg.thinking == "reasoning"
    assert msg.text == "answer"
    assert msg.output_tokens == 20  # 50 - 30
    assert msg.thinking_tokens == 30


# ---------------------------------------------------------------------------
# factory behavior
# ---------------------------------------------------------------------------


def test_factory_builds_mock_with_scenario():
    provider = build_provider(AgentConfig(backend="mock", mock_scenario="happy_path"))
    assert isinstance(provider, MockProvider)


@pytest.mark.parametrize("backend,var", [
    ("nvidia", "nvidia_api_key"),
    ("anthropic", "ANTHROPIC_API_KEY"),
    ("openai", "OPENAI_API_KEY"),
])
def test_factory_missing_keys_name_the_env_var(monkeypatch, backend, var):
    monkeypatch.delenv(var, raising=False)
    with pytest.raises(ProviderError, match=var):
        build_provider(AgentConfig(backend=backend))


def test_factory_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Valid backends"):
        build_provider(AgentConfig(backend="wat"))
    assert set(BACKENDS) == {"mock", "anthropic", "openai", "ollama", "nvidia"}


def test_factory_ollama_unreachable_gives_actionable_error(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:9/v1")  # nothing listens
    with pytest.raises(ProviderError, match="[Oo]llama model"):
        build_provider(AgentConfig(backend="ollama", model="no-such-model-xyz"))
