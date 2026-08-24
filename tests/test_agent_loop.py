import pytest

from codebase_create.agent_loop import STUCK_STREAK_LIMIT, run_agent
from codebase_create.config import AgentConfig
from codebase_create.executor import TempWorkspace
from codebase_create.models import (
    AssistantMessage,
    RunFinished,
    ToolCall,
    TurnStarted,
)
from codebase_create.providers.base import Provider
from codebase_create.providers.mock import MockProvider


def make_config(**overrides) -> AgentConfig:
    return AgentConfig(backend="mock", sandbox="subprocess", **overrides)


def run_scenario(scenario: str, on_event=None, workspace=None, **overrides):
    config = make_config(mock_scenario=scenario, **overrides)
    provider = MockProvider(scenario)
    return run_agent(
        "Build factorial.", provider, config,
        on_event=on_event, workspace=workspace,
    )


# ---------------------------------------------------------------------------
# scenario end-to-end behavior through the real loop
# ---------------------------------------------------------------------------


def test_happy_path_succeeds_with_verified_files():
    ws = TempWorkspace()
    try:
        report = run_scenario("happy_path", workspace=ws)
        assert report.success
        assert report.turns_used == 2  # write turn + verified run turn
        assert report.failure_category == "none"
        assert (ws.path / "solution.py").exists()
        assert (ws.path / "test_solution.py").exists()
    finally:
        ws.cleanup()


def test_fix_after_failure_shows_failure_then_pass_arc():
    report = run_scenario("fix_after_failure")
    assert report.success
    verdicts = [r.content.splitlines()[0]
                for t in report.turns
                for r in t.tool_results if r.name == "run_tests"]
    assert verdicts[0].startswith("FAILED")
    assert verdicts[-1] == "PASSED: 3 test(s) passed."


def test_bad_tool_args_survives_dispatcher_errors():
    report = run_scenario("bad_tool_args")
    assert report.success
    errors = [r for t in report.turns for r in t.tool_results if r.is_error]
    assert len(errors) == 2  # loop continued past invocation errors


def test_stuck_loop_detector_fires_before_script_exhaustion():
    report = run_scenario("stuck_loop")
    assert not report.success
    assert report.failure_category == "stuck_loop"
    # The script holds 5 turns; the detector must stop it after the third
    # identical verdict instead of exhausting the script.
    assert report.turns_used == 1 + STUCK_STREAK_LIMIT
    assert "Repeated identical failure" in report.failure_summary


def test_premature_finish_is_nudged_and_recovers():
    report = run_scenario("premature_finish")
    assert report.success
    assert report.turns_used == 3  # finish attempt + writes + verified run
    nudge_replies = [t.assistant_text for t in report.turns[:2]]
    assert "obviously correct" in nudge_replies[0]


# ---------------------------------------------------------------------------
# termination paths needing a custom provider
# ---------------------------------------------------------------------------


class AlternatingFailingProvider(Provider):
    """Distinct failing variant every turn: never trips the stuck detector."""

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, system_prompt, messages, tools, temperature=0.1):
        self.calls += 1
        return AssistantMessage(
            text=f"variant {self.calls}",
            tool_calls=[
                ToolCall(id=f"{self.calls}a", name="write_file", arguments={
                    "path": "solution.py",
                    "content": "def factorial(n):\n    return 720\n",
                }),
                ToolCall(id=f"{self.calls}b", name="write_file", arguments={
                    "path": "test_solution.py",
                    "content": (
                        "from solution import factorial\n"
                        f"def test_x():\n    assert factorial(5) == {100 + self.calls}\n"
                    ),
                }),
                ToolCall(id=f"{self.calls}c", name="run_tests", arguments={}),
            ],
            input_tokens=100 * self.calls,
            output_tokens=10,
        )


def test_turn_budget_exhaustion_with_distinct_failures():
    report = run_agent(
        "impossible task",
        AlternatingFailingProvider(),
        make_config(max_turns=4),
    )
    assert not report.success
    assert report.failure_category == "turn_budget_exhausted"
    assert report.turns_used == 4


def test_token_accounting_accumulates():
    report = run_agent(
        "task",
        AlternatingFailingProvider(),
        make_config(max_turns=3),
    )
    assert report.total_input_tokens == 100 + 200 + 300
    assert report.total_output_tokens == 30


# ---------------------------------------------------------------------------
# event stream contract
# ---------------------------------------------------------------------------


def test_event_stream_order_and_counts():
    events = []
    report = run_scenario("happy_path", on_event=events.append)

    assert isinstance(events[0], TurnStarted)
    assert isinstance(events[-1], RunFinished) and events[-1].success

    kinds = [type(e).__name__ for e in events]
    assert kinds.count("ToolCalled") == 3      # two writes + one run_tests
    assert kinds.count("ObservationReady") == 3
    assert kinds.count("TurnStarted") == report.turns_used
    # every ToolCalled is immediately followed by its ObservationReady
    for i, kind in enumerate(kinds):
        if kind == "ToolCalled":
            assert kinds[i + 1] == "ObservationReady"


# ---------------------------------------------------------------------------
# workspace ownership + determinism
# ---------------------------------------------------------------------------


def test_injected_workspace_survives_run():
    ws = TempWorkspace()
    try:
        report = run_scenario("happy_path", workspace=ws)
        assert report.success
        assert ws.path.exists(), "caller-owned workspace must not be cleaned"
    finally:
        ws.cleanup()


def test_runs_are_deterministic_modulo_timing():
    def snapshot():
        r = run_scenario("fix_after_failure")
        return {
            "success": r.success,
            "turns_used": r.turns_used,
            "category": r.failure_category,
            "tokens": (r.total_input_tokens, r.total_output_tokens),
            "calls": [[c.name for c in t.tool_calls] for t in r.turns],
        }

    assert snapshot() == snapshot()
