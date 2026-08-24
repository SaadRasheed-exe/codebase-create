"""Terminal renderers consuming the agent event stream.

Renderers are pure consumers: they never touch the provider, the
dispatcher, or the workspace. The same event stream that drives a Rich
experience on an interactive terminal drives plain log lines in a pipe.

No spinners here by design: wrapping provider.complete() from a
callback means cross-callback status contexts, which are fragile.
A Textual dashboard (future phase) owns live views properly.
"""

import json
import sys
from typing import Protocol

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

from codebase_create.models import (
    AgentEvent,
    AgentRunReport,
    AssistantReplied,
    ObservationReady,
    ToolCalled,
    TurnStarted,
)


MAX_ARG_PREVIEW = 60
ERROR_LINES_SHOWN = 6


def _preview_arguments(arguments: dict) -> str:
    parts = []
    for key, value in arguments.items():
        text = value if isinstance(value, str) else json.dumps(value)
        if len(text) > MAX_ARG_PREVIEW:
            text = text[: MAX_ARG_PREVIEW - 3] + "..."
        parts.append(f"{key}={text!r}")
    return ", ".join(parts)


class Renderer(Protocol):
    def handle_event(self, event: AgentEvent) -> None: ...
    def render_report(self, report: AgentRunReport) -> None: ...


class PlainRenderer:
    """Log-style output safe for pipes, CI logs, and --json pairing."""

    def __init__(self, stdout=None) -> None:
        self._out = stdout if stdout is not None else sys.stdout

    def _emit(self, text: str) -> None:
        print(text, file=self._out)

    def handle_event(self, event: AgentEvent) -> None:
        if isinstance(event, TurnStarted):
            self._emit(f"-- turn {event.turn_index} --")
        elif isinstance(event, AssistantReplied):
            for line in event.text.splitlines() or [""]:
                self._emit(f"  {line}")
        elif isinstance(event, ToolCalled):
            self._emit(f"> {event.record.name}({_preview_arguments(event.record.arguments)})")
        elif isinstance(event, ObservationReady):
            result = event.result
            first_line = result.content.splitlines()[0] if result.content else ""
            if result.is_error:
                marker = "!"
            elif result.success:
                marker = "+"
            else:
                marker = "-"
            self._emit(f"<{marker} {first_line}")

    def render_report(self, report: AgentRunReport) -> None:
        self._emit("-" * 60)
        status = "SUCCESS" if report.success else f"FAILURE ({report.failure_category})"
        self._emit(f"{status}: {report.failure_summary}")
        self._emit(
            f"Turns used: {report.turns_used}/{report.max_turns}, "
            f"tokens in/out: {report.total_input_tokens}/{report.total_output_tokens}"
        )
        for path in report.files:
            self._emit(f"file: {path}")


class RichRenderer:
    """Color-coded interactive experience."""

    def __init__(self, console: Console | None = None) -> None:
        self.console = console if console is not None else Console()

    def handle_event(self, event: AgentEvent) -> None:
        c = self.console
        if isinstance(event, TurnStarted):
            c.rule(f"turn {event.turn_index}", style="dim")
        elif isinstance(event, AssistantReplied):
            if event.text.strip():
                c.print(Text(event.text, style="italic dim"))
        elif isinstance(event, ToolCalled):
            args = _preview_arguments(event.record.arguments)
            c.print(f"[cyan]> {event.record.name}[/cyan]({args})")
        elif isinstance(event, ObservationReady):
            result = event.result
            lines = result.content.splitlines()
            if result.is_error:
                body = "\n".join(lines[:ERROR_LINES_SHOWN])
                c.print(Panel(body, title="error", border_style="red"))
            else:
                style = "green" if result.success else "yellow"
                marker = "+" if result.success else "-"
                c.print(f"[{style}]{marker} {lines[0] if lines else ''}[/{style}]")

    def render_report(self, report: AgentRunReport) -> None:
        c = self.console
        if report.success:
            headline = Text(
                f"SUCCESS - {report.turns_used}/{report.max_turns} turns, "
                f"{report.total_input_tokens} in / {report.total_output_tokens} out tokens",
                style="bold green",
            )
        else:
            headline = Text(
                f"FAILURE ({report.failure_category}) - "
                f"{report.failure_summary}",
                style="bold red",
            )
        c.print(Panel(headline, title="agent run"))

        if report.files:
            c.rule("final files", style="dim")
            listing = ", ".join(report.files)
            c.print(f"[dim]{listing}[/dim]")
            for path, content in report.files.items():
                c.rule(path, style="bright_black")
                if path.endswith(".py"):
                    c.print(Syntax(content, "python", line_numbers=False))
                else:
                    c.print(content)


def get_renderer(name: str) -> Renderer:
    if name == "rich":
        return RichRenderer()
    if name == "plain":
        return PlainRenderer()
    if name == "auto":
        return RichRenderer() if sys.stdout.isatty() else PlainRenderer()
    raise ValueError(f"Unknown UI mode '{name}'. Valid: rich, plain, auto")
