"""The agentic loop: a model-toolbox conversation that ends in verified code.

Replaces the legacy generate→repair pipeline. Instead of regenerating
artifacts from string templates, the model drives real tools against a
real workspace; every observation it receives comes from actually
running code.

Termination policy (checked in this order):
1. A run_tests observation reports success  -> done, success.
2. Three consecutive identical failure fingerprints -> stuck_loop abort.
3. Text-only reply -> one nudge if nothing was ever verified, then
   accepted as an unverified finish.
4. Turn budget exhausted -> turn_budget_exhausted.

Temperature is fixed per run: diversity on retries now comes from new
observations, not from sampling noise (the legacy adaptive schedule
existed to compensate for blind regeneration).
"""

import time
from typing import Callable

from codebase_create.config import AgentConfig
from codebase_create.executor import TempWorkspace
from codebase_create.models import (
    AgentEvent,
    AgentRunReport,
    AgentTurn,
    AssistantMessage,
    AssistantReplied,
    FailureCategory,
    ObservationReady,
    RunFinished,
    TextDelta,
    ThinkingDelta,
    ToolCall,
    ToolCalled,
    ToolResult,
    ToolResultMessage,
    TurnStarted,
    UserMessage,
)
from codebase_create.prompts import AGENT_SYSTEM_PROMPT
from codebase_create.providers.base import Provider, ProviderError
from codebase_create.tools import MAX_READ_CHARS, TOOL_SPECS, ToolDispatcher


SNAPSHOT_MAX_CHARS = MAX_READ_CHARS


def _snapshot_files(ws: TempWorkspace) -> dict[str, str]:
    """Capture the final workspace contents for the report."""
    files: dict[str, str] = {}
    for info in ws.list_files():
        try:
            content = ws.read_file(info.path)
        except Exception:
            continue  # unreadable file: omit rather than fail the report
        if len(content) > SNAPSHOT_MAX_CHARS:
            hidden = len(content) - SNAPSHOT_MAX_CHARS
            content = content[:SNAPSHOT_MAX_CHARS] + f"\n... [truncated {hidden} chars]"
        files[info.path] = content
    return files


NUDGE_MESSAGE = (
    "You ended your turn without tool calls, but no run_tests call has "
    "passed yet, so there is no verified solution. Either continue "
    "working (write files, run tests) or explain why the task cannot "
    "be completed."
)

STUCK_STREAK_LIMIT = 3

EventCallback = Callable[[AgentEvent], None]


def _failing_run_fingerprint(
    results: list[ToolResult],
) -> tuple[str, str] | None:
    """Identity of the last failing run_tests observation in this batch.

    None when the batch contained no failing test run. The fingerprint
    combines the verdict header with the first failure line so that two
    runs only count as 'the same failure' when both the counts and the
    leading message agree.
    """
    last = None
    for result in results:
        if result.name == "run_tests" and not result.is_error and not result.success:
            last = result
    if last is None:
        return None
    lines = last.content.splitlines()
    header = lines[0] if lines else ""
    # Failure bodies span many physical lines, so compare a flattened
    # prefix rather than only the first line after the verdict header.
    flat = " ".join(last.content.split())
    return (header, flat[:200])


def run_agent(
    user_request: str,
    provider: Provider,
    config: AgentConfig,
    on_event: EventCallback | None = None,
    workspace: TempWorkspace | None = None,
) -> AgentRunReport:
    """Run the loop until verified success or a termination condition.

    An injected ``workspace`` is owned by the caller and never cleaned
    up (REPL sessions reuse it across requests); otherwise the loop
    creates its own workspace honoring config.keep_artifacts.
    """
    emit: EventCallback = on_event or (lambda event: None)
    owns_workspace = workspace is None
    ws = workspace or TempWorkspace(keep_artifacts=config.keep_artifacts)
    dispatcher = ToolDispatcher(ws, config)

    def _stream_handler(kind: str, delta: str) -> bool:
        nonlocal _thinking_chars
        if kind == "thinking":
            _thinking_chars += len(delta)
            if _thinking_chars > config.max_thinking_tokens_per_turn * 4:
                return False  # budget exceeded — tell provider to stop
            emit(ThinkingDelta(text=delta))
        elif kind == "text":
            emit(TextDelta(text=delta))
        return True

    messages: list = [UserMessage(user_request)]
    turns: list[AgentTurn] = []
    nudges_used = 0
    streak_fingerprint: tuple[str, str] | None = None
    streak_length = 0
    _thinking_chars = 0

    def report(success: bool, category: FailureCategory, summary: str) -> AgentRunReport:
        finished = AgentRunReport(
            success=success,
            turns_used=len(turns),
            max_turns=config.max_turns,
            failure_category="none" if success else category,
            failure_summary="All tests passed" if success else summary,
            turns=turns,
            total_input_tokens=sum(t.input_tokens for t in turns),
            total_output_tokens=sum(t.output_tokens for t in turns),
            total_thinking_tokens=sum(t.thinking_tokens for t in turns),
            files=_snapshot_files(ws),
        )
        emit(RunFinished(success=finished.success, turns_used=finished.turns_used))
        if owns_workspace:
            ws.cleanup()
        return finished

    try:
        for index in range(1, config.max_turns + 1):
            emit(TurnStarted(turn_index=index))
            start = time.perf_counter()
            _thinking_chars = 0

            reply = provider.complete(
                system_prompt=AGENT_SYSTEM_PROMPT,
                messages=messages,
                tools=TOOL_SPECS,
                temperature=config.generation_temperature,
                on_delta=_stream_handler if config.stream_output else None,
            )
            turn = AgentTurn(
                index=index,
                assistant_text=reply.text,
                thinking_text=reply.thinking,
                input_tokens=reply.input_tokens,
                output_tokens=reply.output_tokens,
                thinking_tokens=reply.thinking_tokens,
            )

            # --- finish attempt: text-only turn -----------------------
            if not reply.tool_calls:
                turn.duration_sec = time.perf_counter() - start
                turns.append(turn)
                if nudges_used == 0:
                    nudges_used += 1
                    messages.append(AssistantMessage(text=reply.text))
                    messages.append(UserMessage(NUDGE_MESSAGE))
                    continue
                summary = reply.text.strip()[:300] or "Model finished without verification"
                return report(False, "no_verified_solution", f"Unverified finish: {summary}")

            # --- execute the requested tools --------------------------
            if reply.text or reply.thinking:
                emit(AssistantReplied(text=reply.text, thinking=reply.thinking))
            results: list[ToolResult] = []
            for call in reply.tool_calls:
                emit(ToolCalled(record=call))
                result = dispatcher.execute(call)
                emit(ObservationReady(result=result))
                results.append(result)

            turn.tool_calls = list(reply.tool_calls)
            turn.tool_results = results
            turn.duration_sec = time.perf_counter() - start
            turns.append(turn)

            messages.append(
                AssistantMessage(
                    text=reply.text,
                    tool_calls=reply.tool_calls,
                    input_tokens=reply.input_tokens,
                    output_tokens=reply.output_tokens,
                )
            )
            messages.append(ToolResultMessage(results=results))

            # --- exit 1: verified success -----------------------------
            if any(result.success for result in results):
                return report(True, "none", "All tests passed")

            # --- exit 2: repeated identical failures -------------------
            fingerprint = _failing_run_fingerprint(results)
            if fingerprint == streak_fingerprint and fingerprint is not None:
                streak_length += 1
            else:
                streak_fingerprint = fingerprint
                streak_length = 1
            if streak_length >= STUCK_STREAK_LIMIT:
                header = fingerprint[0]
                return report(
                    False,
                    "stuck_loop",
                    f"Repeated identical failure {STUCK_STREAK_LIMIT}x: {header}",
                )

        # --- exit 4: budget exhausted ---------------------------------
        last_run = next(
            (r.content.splitlines()[0]
             for t in reversed(turns)
             for r in reversed(t.tool_results)
             if r.name == "run_tests"),
            "no test run recorded",
        )
        return report(False, "turn_budget_exhausted", f"{config.max_turns} turns used; last verdict: {last_run}")

    except ProviderError as ex:
        return report(False, "provider_error", str(ex))
