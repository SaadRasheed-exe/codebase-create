import json

import pytest
from rich.console import Console

from codebase_create.agent_loop import run_agent
from codebase_create.app import main as app_main
from codebase_create.config import AgentConfig
from codebase_create.executor import TempWorkspace
from codebase_create.models import (
    AssistantReplied,
    ObservationReady,
    RunFinished,
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
        "--backend", "mock", "--sandbox", "subprocess",
        "--ui", "plain",
    ])
    assert code == 0


def test_cli_json_output_is_machine_readable(capsys):
    code = app_main([
        "Build a factorial function.",
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
    code = app_main(["task", "--backend", "nvidia", "--ui", "plain"])
    assert code == 2
