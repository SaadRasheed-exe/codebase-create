"""Terminal renderers consuming the agent event stream.

Renderers are pure consumers: they never touch the provider, the
dispatcher, or the workspace. The same event stream that drives a Rich
experience on an interactive terminal drives plain log lines in a pipe.
"""

import json
import sys
from typing import Protocol

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

from codebase_create.models import (
    AgentEvent,
    AgentRunReport,
    AssistantReplied,
    ObservationReady,
    TextDelta,
    ThinkingDelta,
    ToolCalled,
    TurnStarted,
)


MAX_ARG_PREVIEW = 60
ERROR_LINES_SHOWN = 6
THINKING_LIVE_TAIL_LINES = 10


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

    def __init__(self, stdout=None, show_thinking: bool = False) -> None:
        self._out = stdout if stdout is not None else sys.stdout
        self._show_thinking = show_thinking
        self._in_thinking = False

    def _emit(self, text: str, end: str = "\n") -> None:
        print(text, end=end, file=self._out)

    def handle_event(self, event: AgentEvent) -> None:
        if isinstance(event, TurnStarted):
            self._emit(f"-- turn {event.turn_index} --")
        elif isinstance(event, ThinkingDelta):
            if self._show_thinking:
                self._emit(event.text, end="")
        elif isinstance(event, TextDelta):
            self._emit(event.text, end="")
        elif isinstance(event, AssistantReplied):
            # Close thinking block if streaming left it open
            if self._in_thinking:
                self._emit("")
                self._in_thinking = False
            if event.thinking:
                if self._show_thinking:
                    for line in event.thinking.splitlines() or [""]:
                        self._emit(f"# {line}")
                else:
                    self._emit("  [thinking hidden — use --thinking to show]")
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
    """Color-coded interactive experience with streaming thinking panel."""

    def __init__(self, console: Console | None = None, show_thinking: bool = False) -> None:
        self.console = console if console is not None else Console()
        self._show_thinking = show_thinking
        self._in_thinking = False
        self._live: Live | None = None
        self._thinking_text = ""

    def _close_thinking(self) -> None:
        if self._in_thinking:
            if self._live is not None:
                self._live.stop()
                self._live = None
            if self._show_thinking and self._thinking_text:
                self._print_thinking_block(self._thinking_text)
            self._thinking_text = ""
            self._in_thinking = False

    def _print_thinking_block(self, text: str) -> None:
        """Print the full thinking as plain, scrollable lines.

        A Rich Panel is rendered as a single compound shape that Rich
        clips to the terminal height (the 'three dots' symptom).  Plain
        text lines go into the terminal scrollback, so the whole
        reasoning trace can be scrolled through.
        """
        c = self.console
        c.rule("thinking", style="dim")
        for line in text.splitlines() or [""]:
            c.print(Text(f"  {line}", style="dim"))

    def _live_panel(self) -> Panel:
        """Live preview shows only the tail of thinking while streaming."""
        lines = self._thinking_text.splitlines()
        body = "\n".join(lines[-THINKING_LIVE_TAIL_LINES:]) if lines else ""
        return Panel(body, title="thinking", border_style="dim")

    def _start_live(self) -> None:
        self._in_thinking = True
        self._thinking_text = ""
        self._live = Live(
            self._live_panel(),
            console=self.console,
            auto_refresh=True,
            refresh_per_second=8,
        )
        self._live.start()

    def handle_event(self, event: AgentEvent) -> None:
        c = self.console
        if isinstance(event, TurnStarted):
            self._close_thinking()
            c.rule(f"turn {event.turn_index}", style="dim")
        elif isinstance(event, ThinkingDelta):
            if self._show_thinking:
                if not self._in_thinking:
                    self._start_live()
                self._thinking_text += event.text
                if self._live is not None:
                    self._live.update(self._live_panel())
            else:
                if not self._in_thinking:
                    c.print(
                        Text("[thinking hidden — use --thinking to show]", style="dim italic"),
                        end="",
                    )
                    self._in_thinking = True  # suppress repeated hints
        elif isinstance(event, TextDelta):
            self._close_thinking()
            c.print(event.text, end="")
        elif isinstance(event, AssistantReplied):
            already_shown = self._in_thinking
            self._close_thinking()
            if event.thinking:
                if self._show_thinking:
                    if not already_shown:
                        self._print_thinking_block(event.thinking)
                else:
                    if not already_shown:
                        c.print(Text("[thinking hidden — use --thinking to show]", style="dim italic"))
            if event.text.strip():
                c.print(Text(event.text, style="italic dim"))
        elif isinstance(event, ToolCalled):
            self._close_thinking()
            args = _preview_arguments(event.record.arguments)
            c.print(f"[cyan]> {event.record.name}[/cyan]({args})")
        elif isinstance(event, ObservationReady):
            self._close_thinking()
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
        self._close_thinking()
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


def get_renderer(name: str, show_thinking: bool = False) -> Renderer:
    if name == "rich":
        return RichRenderer(show_thinking=show_thinking)
    if name == "plain":
        return PlainRenderer(show_thinking=show_thinking)
    if name == "auto":
        return RichRenderer(show_thinking=show_thinking) if sys.stdout.isatty() else PlainRenderer(show_thinking=show_thinking)
    raise ValueError(f"Unknown UI mode '{name}'. Valid: rich, plain, auto")
