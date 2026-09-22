import json

import pytest
from rich.console import Console

from codebase_create.agent_loop import run_agent
from codebase_create.app import main as app_main
from codebase_create.config import AgentConfig
from codebase_create.executor import TempWorkspace
from codebase_create.models import (
    AssistantMessage,
    AssistantReplied,
    ObservationReady,
    RunFinished,
    TextDelta,
    ThinkingDelta,
    ToolCall,
    ToolCalled,
    ToolResult,
    TurnStarted,
)
from codebase_create.providers import build_provider
from codebase_create.repl import ReplDriver
from codebase_create.ui import PlainRenderer, RichRenderer, get_renderer


# ---------------------------------------------------------------------------
# renderers on synthetic event streams
# ---------------------------------------------------------------------------


def sample_events():
    return [
        TurnStarted(turn_index=1),
        AssistantReplied(text="Writing files."),
        ToolCalled(record=ToolCall(id="1", name="write_file",
                                   arguments={"path": "a.py", "content": "x = 1\n" * 30})),
        ObservationReady(result=ToolResult(call_id="1", name="write_file", content="Wrote 60 bytes")),
        ObservationReady(result=ToolResult(call_id="2", name="run_tests",
                                           content="FAILED (test_failure): 1 failed, 0 errors, 0 passed")),
        ObservationReady(result=ToolResult(call_id="3", name="run_tests",
                                           content="boom", is_error=True)),
        RunFinished(success=False, turns_used=1),
    ]


def test_plain_renderer_marks_verdicts_and_errors(capsys):
    renderer = PlainRenderer()
    for event in sample_events():
        renderer.handle_event(event)

    out = capsys.readouterr().out
    assert "-- turn 1 --" in out
    assert "> write_file(" in out
    assert "<- FAILED" in out      # failing observation marker
    assert "<! boom" in out        # invocation error marker
    # RunFinished is deliberately not rendered by handle_event
    assert len(out.splitlines()) == 6


def test_rich_renderer_renders_tool_names_and_errors():
    console = Console(record=True, width=100, force_terminal=False)
    renderer = RichRenderer(console)
    for event in sample_events():
        renderer.handle_event(event)
    text = console.export_text()
    assert "write_file" in text
    assert "error" in text


def test_get_renderer_auto_and_validation(monkeypatch):
    monkeypatch.setattr("sys.stdout.isatty", lambda: True, raising=False)
    assert isinstance(get_renderer("auto"), RichRenderer)
    assert isinstance(get_renderer("plain"), PlainRenderer)
    with pytest.raises(ValueError, match="Unknown UI mode"):
        get_renderer("curses")


# ---------------------------------------------------------------------------
# thinking display


def test_plain_renderer_thinking_hidden(capsys):
    renderer = PlainRenderer(show_thinking=False)
    renderer.handle_event(AssistantReplied(text="done", thinking="step 1: think"))
    out = capsys.readouterr().out
    assert "[thinking hidden" in out
    assert "step 1: think" not in out
    assert "done" in out


def test_plain_renderer_thinking_shown(capsys):
    renderer = PlainRenderer(show_thinking=True)
    renderer.handle_event(AssistantReplied(text="done", thinking="step 1: think"))
    out = capsys.readouterr().out
    assert "# step 1: think" in out
    assert "done" in out
    assert "[thinking hidden" not in out


def test_rich_renderer_thinking_hidden():
    console = Console(record=True, width=100, force_terminal=False)
    renderer = RichRenderer(console, show_thinking=False)
    renderer.handle_event(AssistantReplied(text="done", thinking="reasoning here"))
    text = console.export_text()
    assert "reasoning here" not in text
    assert "[thinking hidden" in text


def test_rich_renderer_thinking_shown():
    console = Console(record=True, width=100, force_terminal=False)
    renderer = RichRenderer(console, show_thinking=True)
    renderer.handle_event(AssistantReplied(text="done", thinking="reasoning here"))
    text = console.export_text()
    assert "reasoning here" in text


# ---------------------------------------------------------------------------
# streaming display


def test_plain_renderer_streams_thinking(capsys):
    renderer = PlainRenderer(show_thinking=True)
    renderer.handle_event(ThinkingDelta(text="step "))
    renderer.handle_event(ThinkingDelta(text="one"))
    renderer.handle_event(TextDelta(text="hello"))
    out = capsys.readouterr().out
    assert "step one" in out
    assert "hello" in out


def test_plain_renderer_streams_hidden_thinking(capsys):
    renderer = PlainRenderer(show_thinking=False)
    renderer.handle_event(ThinkingDelta(text="secret"))
    renderer.handle_event(TextDelta(text="visible"))
    out = capsys.readouterr().out
    assert "secret" not in out
    assert "visible" in out


def test_rich_renderer_streams_thinking_in_panel():
    console = Console(record=True, width=80, force_terminal=False)
    renderer = RichRenderer(console, show_thinking=True)
    renderer.handle_event(ThinkingDelta(text="reasoning"))
    renderer.handle_event(TextDelta(text="answer"))
    text = console.export_text()
    assert "thinking" in text  # panel header
    assert "reasoning" in text
    assert "answer" in text


def test_rich_renderer_streams_hidden_thinking():
    console = Console(record=True, width=80, force_terminal=False)
    renderer = RichRenderer(console, show_thinking=False)
    renderer.handle_event(ThinkingDelta(text="secret"))
    renderer.handle_event(TextDelta(text="visible"))
    text = console.export_text()
    assert "secret" not in text
    assert "visible" in text


# ---------------------------------------------------------------------------
# thinking budget enforcement


def test_thinking_budget_stops_streaming():
    """When streaming thinking exceeds the budget, the stream is cut and a
    truncation marker is left in its place."""
    events: list = []

    def collect(event):
        events.append(event)

    config = AgentConfig(
        backend="mock", sandbox="subprocess",
        max_thinking_tokens_per_turn=5,  # 5 tokens -> 20 chars
    )

    class StreamingThinkingProvider:
        """Mimics a streaming provider honoring on_delta's return value.
        Never terminates successfully: text-only reply ends the loop."""

        def __init__(self):
            self.calls = 0

        def complete(self, system_prompt, messages, tools, temperature=0.1, on_delta=None):
            self.calls += 1
            if on_delta is not None:
                for _ in range(200):
                    if not on_delta("thinking", "x"):
                        break
            return AssistantMessage(text=f"done {self.calls}", thinking="x" * 200)

    report = run_agent(
        "task", StreamingThinkingProvider(), config, on_event=collect
    )
    # First call streams exactly the budgeted chars before the marker.
    # Second call repeats (fresh budget per turn), then text-only ends it.
    x_chars = sum(
        len(e.text) for e in events
        if isinstance(e, ThinkingDelta) and e.text == "x"
    )
    markers = [
        e.text for e in events
        if isinstance(e, ThinkingDelta) and "thinking truncated" in e.text
    ]
    assert x_chars == 2 * (5 * 4)
    assert len(markers) == 2
    assert "max 5 tokens/turn" in markers[0]
    assert report.success is False


def test_rich_renderer_long_thinking_not_clipped():
    """Every thinking line lands in the output as plain scrollable text,
    even when the total far exceeds one terminal screen."""
    console = Console(record=True, width=60, force_terminal=False)
    renderer = RichRenderer(console, show_thinking=True)
    body = "\n".join(f"line {i:02d}" for i in range(60))  # ~60 lines
    for chunk in body.split():
        renderer.handle_event(ThinkingDelta(text=chunk + " "))
    renderer.handle_event(TextDelta(text="answer"))
    text = console.export_text()
    for i in range(60):
        assert f"line {i:02d}" in text
    assert "thinking" in text  # rule header
    assert "answer" in text


def test_rich_thinking_not_duplicated():
    """Streamed thinking printed once by _close_thinking is not repeated
    by a subsequent AssistantReplied event."""
    console = Console(record=True, width=80, force_terminal=False)
    renderer = RichRenderer(console, show_thinking=True)
    renderer.handle_event(ThinkingDelta(text="step one "))
    renderer.handle_event(AssistantReplied(text="done", thinking="step one "))
    text = console.export_text()
    # The permanent scrollable block line is printed exactly once;
    # AssistantReplied does not re-print it. (The transient Live panel
    # render is preserved separately by the recording console.)
    assert text.count("  step one") == 1
    assert "done" in text


def test_thinking_budget_emits_truncation_marker():
    """The truncation marker text appears in the emitted event stream."""
    events: list = []

    def collect(event):
        events.append(event)

    config = AgentConfig(
        backend="mock", sandbox="subprocess",
        max_thinking_tokens_per_turn=2,  # 2 tokens -> 8 chars
    )

    class StreamingThinkingProvider:
        def __init__(self):
            self.calls = 0

        def complete(self, system_prompt, messages, tools, temperature=0.1, on_delta=None):
            self.calls += 1
            if on_delta is not None:
                for _ in range(100):
                    if not on_delta("thinking", "x"):
                        break
            return AssistantMessage(text=f"done {self.calls}", thinking="x" * 100)

    run_agent("task", StreamingThinkingProvider(), config, on_event=collect)
    marker = next(
        e.text for e in events
        if isinstance(e, ThinkingDelta) and "thinking truncated" in e.text
    )
    assert "thinking truncated" in marker


def test_plain_report_summary_includes_files(capsys):
    report = run_agent(
        "task",
        build_provider(AgentConfig(backend="mock", mock_scenario="happy_path")),
        AgentConfig(backend="mock", sandbox="subprocess"),
    )
    PlainRenderer().render_report(report)
    out = capsys.readouterr().out
    assert "SUCCESS" in out
    assert "file: solution.py" in out


# ---------------------------------------------------------------------------
# workspace snapshot on reports
# ---------------------------------------------------------------------------


def test_report_snapshots_workspace_files():
    ws = TempWorkspace()
    try:
        report = run_agent(
            "task",
            build_provider(AgentConfig(backend="mock", mock_scenario="happy_path")),
            AgentConfig(backend="mock", sandbox="subprocess"),
            workspace=ws,
        )
        assert set(report.files) == {"solution.py", "test_solution.py"}
        assert "def factorial" in report.files["solution.py"]
    finally:
        ws.cleanup()


# ---------------------------------------------------------------------------
# REPL driver grammar (injected IO, no terminal needed)
# ---------------------------------------------------------------------------


class Writer:
    """file-like adapter so PlainRenderer output joins driver output."""

    def __init__(self, sink: list) -> None:
        self._sink = sink

    def write(self, s: str) -> int:
        self._sink.append(s)
        return len(s)


@pytest.fixture()
def repl_outputs():
    outputs: list[str] = []
    return outputs


def make_repl(inputs: list[str], outputs: list[str]) -> ReplDriver:
    config = AgentConfig(backend="mock", mock_scenario="happy_path",
                         sandbox="subprocess")

    def fake_input(_prompt: str) -> str:
        if not inputs:  # end of stream must look like EOF, not IndexError
            raise EOFError
        return inputs.pop(0)

    return ReplDriver(
        config,
        PlainRenderer(stdout=Writer(outputs)),
        input_fn=fake_input,
        output_fn=outputs.append,
    )


def test_repl_full_session_lifecycle(repl_outputs):
    driver = make_repl([
        "Build a factorial function.",
        "/files",
        "/reset",
        "/files",
        "/bogus",
        "/exit",
    ], repl_outputs)

    assert driver.run() == 0
    text = "".join(str(o) for o in repl_outputs)
    assert "SUCCESS" in text                                   # request ran
    assert "solution.py (" in text                             # /files after run
    assert "(workspace is empty)" in text                      # /files after /reset
    assert "Unknown command '/bogus'" in text                  # rejected locally


def test_repl_eof_exits_cleanly(repl_outputs):
    driver = make_repl([], repl_outputs)
    assert driver.run() == 0
    assert "interactive agent mode" in "".join(map(str, repl_outputs))


def test_repl_keyboard_interrupt_at_prompt_continues(repl_outputs):
    calls = {"n": 0}

    def flaky_input(_prompt):
        calls["n"] += 1
        if calls["n"] == 1:
            raise KeyboardInterrupt
        if calls["n"] == 2:
            raise EOFError
        raise AssertionError("unreachable")

    config = AgentConfig(backend="mock", sandbox="subprocess")
    driver = ReplDriver(config, PlainRenderer(stdout=Writer(repl_outputs)),
                        input_fn=flaky_input, output_fn=repl_outputs.append)
    assert driver.run() == 0
    assert "^C" in "".join(str(o) for o in repl_outputs)


# ---------------------------------------------------------------------------
# CLI end-to-end (offline via mock backend)
# ---------------------------------------------------------------------------


def test_cli_one_shot_success_exit_code():
    code = app_main([
        "Build a factorial function.",
        "--onetime",
        "--backend", "mock", "--sandbox", "subprocess",
        "--ui", "plain",
    ])
    assert code == 0


def test_cli_json_output_is_machine_readable(capsys):
    code = app_main([
        "Build a factorial function.",
        "--onetime",
        "--backend", "mock", "--sandbox", "subprocess",
        "--ui", "plain", "--json",
    ])
    assert code == 0
    out = capsys.readouterr().out
    payload = json.loads(out[out.index("{"):])
    assert payload["success"] is True
    assert set(payload["files"]) == {"solution.py", "test_solution.py"}


def test_cli_configuration_error_exit_code_two(monkeypatch):
    monkeypatch.delenv("nvidia_api_key", raising=False)
    code = app_main(["task", "--onetime", "--backend", "nvidia", "--ui", "plain"])
    assert code == 2
