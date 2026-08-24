"""CLI entry point for the agentic coding assistant.

One-shot:   python app.py "Build a palindrome checker."
Offline:    python app.py "..." --backend mock --mock-scenario happy_path
Interactive python app.py            (or --repl)
"""

import argparse
import json
import sys

from codebase_create.agent_loop import run_agent
from codebase_create.config import AgentConfig
from codebase_create.providers import BACKENDS, build_provider
from codebase_create.providers.base import ProviderError
from codebase_create.ui import get_renderer


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "AI coding agent: writes implementation and tests, runs them "
            "in a sandbox, and iterates until green."
        )
    )
    parser.add_argument(
        "prompt", nargs="?", default=None,
        help="Programming request; omit to enter interactive REPL mode",
    )
    parser.add_argument("--backend", choices=BACKENDS, default=None,
                        help="LLM backend (default from AGENT_BACKEND env)")
    parser.add_argument("--model", default=None, help="Model name for the backend")
    parser.add_argument("--mock-scenario", default=None,
                        help="Script replayed by --backend mock")
    parser.add_argument("--max-turns", type=int, default=None,
                        help="Model turn budget per request (default 12)")
    parser.add_argument("--timeout", type=int, default=None,
                        help="Per pytest-run timeout in seconds")
    parser.add_argument("--sandbox", choices=["subprocess", "docker"], default=None)
    parser.add_argument("--keep-artifacts", action="store_true",
                        help="Keep workspace directories after runs")
    parser.add_argument("--ui", choices=["rich", "plain", "auto"], default="auto")
    parser.add_argument("--json", action="store_true",
                        help="Print machine-readable report JSON at the end")
    parser.add_argument("--repl", action="store_true",
                        help="Force interactive mode")
    return parser


def _apply_overrides(config: AgentConfig, args: argparse.Namespace) -> None:
    if args.backend:
        config.backend = args.backend
    if args.model:
        config.model = args.model
    if args.mock_scenario:
        config.mock_scenario = args.mock_scenario
    if args.max_turns:
        config.max_turns = args.max_turns
    if args.timeout:
        config.test_timeout_sec = args.timeout
    if args.sandbox:
        config.sandbox = args.sandbox
    if args.keep_artifacts:
        config.keep_artifacts = True


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = AgentConfig.from_env()
    _apply_overrides(config, args)
    renderer = get_renderer(args.ui)

    if args.repl or args.prompt is None:
        from codebase_create.repl import run_repl  # deferred import
        return run_repl(config, renderer)

    try:
        provider = build_provider(config)
    except ProviderError as ex:
        print(f"error: {ex}", file=sys.stderr)
        return 2  # configuration problem, not a task failure

    report = run_agent(args.prompt, provider, config, on_event=renderer.handle_event)
    renderer.render_report(report)

    if args.json:
        print(json.dumps({
            "success": report.success,
            "turns_used": report.turns_used,
            "max_turns": report.max_turns,
            "failure_category": report.failure_category,
            "failure_summary": report.failure_summary,
            "total_input_tokens": report.total_input_tokens,
            "total_output_tokens": report.total_output_tokens,
            "files": report.files,
        }, indent=2))

    return 0 if report.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
