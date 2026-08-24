"""The toolbox exposed to the model.

Two artifacts live here:

- ``TOOL_SPECS``: provider-neutral tool definitions. Providers translate
  these into their native formats in Phase 3; the JSON Schema shape is
  common to OpenAI and Anthropic tool calling.
- ``ToolDispatcher``: executes normalized ``ToolCall`` objects against a
  workspace and never raises — every failure mode becomes an error
  ``ToolResult`` so a confused model cannot crash the loop.

Error semantics are deliberate: ``is_error=True`` is reserved for
invocation-level problems (unknown tool, bad arguments, workspace
violations). Test failures are *observations*, not errors — the model
must receive them as normal feedback it can act on.
"""

from dataclasses import dataclass

from codebase_create.config import AgentConfig
from codebase_create.executor import TempWorkspace
from codebase_create.executor import run_pytest
from codebase_create.models import TestExecutionResult, ToolCall, ToolResult
from codebase_create.test_results import parse_test_result


MAX_READ_CHARS = 20_000
MAX_FAILURE_MESSAGES = 8
MAX_MESSAGE_CHARS = 600


@dataclass(slots=True)
class ToolSpec:
    """A tool definition in provider-neutral form."""

    name: str
    description: str
    parameters: dict


TOOL_SPECS: list[ToolSpec] = [
    ToolSpec(
        name="write_file",
        description=(
            "Create or overwrite a text file in the workspace. "
            "Parent directories are created automatically."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Workspace-relative path, e.g. 'pkg/utils.py'.",
                },
                "content": {
                    "type": "string",
                    "description": "The complete file content.",
                },
            },
            "required": ["path", "content"],
        },
    ),
    ToolSpec(
        name="read_file",
        description="Return the content of a workspace file.",
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Workspace-relative path.",
                },
            },
            "required": ["path"],
        },
    ),
    ToolSpec(
        name="list_files",
        description=(
            "List all files in the workspace with their sizes. "
            "Use this before reading files when unsure what exists."
        ),
        parameters={"type": "object", "properties": {}},
    ),
    ToolSpec(
        name="run_tests",
        description=(
            "Run pytest on the workspace and return a compact verdict "
            "(counts plus failure messages). Call this after writing "
            "implementation and test files."
        ),
        parameters={"type": "object", "properties": {}},
    ),
]


def _require_str(args: dict, key: str) -> str:
    value = args.get(key)
    if not isinstance(value, str):
        raise ValueError(f"Missing or non-string argument '{key}'")
    return value


class ToolDispatcher:
    """Executes model-issued tool calls against an agent workspace."""

    def __init__(self, workspace: TempWorkspace, config: AgentConfig) -> None:
        self._workspace = workspace
        self._config = config

    def execute(self, call: ToolCall) -> ToolResult:
        handlers = {
            "write_file": self._write_file,
            "read_file": self._read_file,
            "list_files": self._list_files,
            "run_tests": self._run_tests,
        }
        if call.name not in handlers:
            available = ", ".join(sorted(spec.name for spec in TOOL_SPECS))
            return ToolResult(
                call_id=call.id,
                name=call.name,
                content=f"Unknown tool '{call.name}'. Available tools: {available}",
                is_error=True,
            )
        if not isinstance(call.arguments, dict):
            return ToolResult(
                call_id=call.id,
                name=call.name,
                content="Tool arguments must be a JSON object",
                is_error=True,
            )
        try:
            content = handlers[call.name](call.arguments)
        except Exception as ex:
            return ToolResult(
                call_id=call.id,
                name=call.name,
                content=f"{type(ex).__name__}: {ex}",
                is_error=True,
            )
        return ToolResult(call_id=call.id, name=call.name, content=content)

    def _write_file(self, args: dict) -> str:
        path = _require_str(args, "path")
        content = _require_str(args, "content")
        self._workspace.write_file(path, content)
        size = len(content.encode("utf-8"))
        return f"Wrote {size} bytes to {path}"

    def _read_file(self, args: dict) -> str:
        path = _require_str(args, "path")
        text = self._workspace.read_file(path)
        if len(text) > MAX_READ_CHARS:
            hidden = len(text) - MAX_READ_CHARS
            return text[:MAX_READ_CHARS] + f"\n... [truncated {hidden} chars]"
        return text

    def _list_files(self, args: dict) -> str:
        infos = self._workspace.list_files()
        if not infos:
            return "(workspace is empty)"
        return "\n".join(f"{info.path} ({info.bytes} bytes)" for info in infos)

    def _run_tests(self, args: dict) -> str:
        junit = self._workspace.path / "results.xml"
        completed = run_pytest(
            work_dir=self._workspace.path,
            junit_file=junit,
            timeout_sec=self._config.test_timeout_sec,
            config=self._config,
        )
        if completed is None:
            result = parse_test_result("", "", junit, timed_out=True)
        else:
            result = parse_test_result(
                completed.stdout,
                completed.stderr,
                junit,
                timed_out=False,
                exit_code=completed.returncode,
            )
        return format_test_observation(result)


def format_test_observation(result: TestExecutionResult) -> str:
    """Compact, token-budgeted summary of a pytest run for the model."""
    if result.timed_out:
        return "TIMEOUT: test execution exceeded the time limit."
    if result.success:
        return f"PASSED: {result.passed} test(s) passed."

    lines = [
        f"FAILED ({result.category}): "
        f"{result.failed} failed, {result.errors} errors, {result.passed} passed"
    ]
    shown = result.failure_messages[:MAX_FAILURE_MESSAGES]
    for i, message in enumerate(shown, start=1):
        message = message.strip()
        if len(message) > MAX_MESSAGE_CHARS:
            message = message[:MAX_MESSAGE_CHARS] + "..."
        lines.append(f"[{i}] {message}")
    hidden = len(result.failure_messages) - len(shown)
    if hidden > 0:
        lines.append(f"... {hidden} more failure message(s) omitted")
    return "\n".join(lines)
