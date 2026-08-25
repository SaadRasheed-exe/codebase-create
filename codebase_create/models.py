from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


FailureCategory = Literal[
    "none",
    "syntax_error",
    "runtime_error",
    "test_failure",
    "timeout",
    "infrastructure_error",
    "stuck_loop",
    "turn_budget_exhausted",
    "provider_error",
    "no_verified_solution",
]


@dataclass(slots=True)
class ExecutionArtifacts:
    work_dir: Path
    solution_file: Path
    test_file: Path
    junit_file: Path


@dataclass(slots=True)
class TestExecutionResult:
    success: bool
    passed: int
    failed: int
    errors: int
    timed_out: bool
    failure_messages: list[str]
    stdout: str
    stderr: str
    category: FailureCategory


# ---------------------------------------------------------------------------
# Agentic building blocks (tool-calling architecture)
#
# Provider-neutral vocabulary: every LLM backend normalizes its native
# tool-calling format (Anthropic tool_use blocks, OpenAI tool_calls JSON
# strings, ...) into ToolCall on the way in, and consumes ToolResult on the
# way back.  The agent loop and UI never see provider-specific shapes.
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ToolCall:
    """A single tool invocation requested by the model."""

    id: str
    name: str
    arguments: dict


@dataclass(slots=True)
class ToolResult:
    """The observation produced by executing a ToolCall."""

    call_id: str
    name: str
    content: str
    is_error: bool = False
    # Structured verdict for verification tools (set by run_tests);
    # None for tools where success is not meaningful.
    success: bool | None = None


@dataclass(slots=True)
class AgentTurn:
    """One round-trip with the model plus everything it triggered."""

    index: int
    assistant_text: str = ""
    thinking_text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    duration_sec: float = 0.0


@dataclass(slots=True)
class FileInfo:
    """A file inside the agent workspace (path is workspace-relative)."""

    path: str
    bytes: int


@dataclass(slots=True)
class ToolSpec:
    """A tool definition in provider-neutral JSON-Schema form."""

    name: str
    description: str
    parameters: dict


# Events emitted by the agent loop.  Renderers subscribe to the same
# stream via a callback; matching is done by type so new renderers can be
# added without touching orchestration logic.


@dataclass(slots=True)
class TurnStarted:
    turn_index: int


@dataclass(slots=True)
class AssistantReplied:
    text: str
    thinking: str = ""


@dataclass(slots=True)
class ToolCalled:
    record: ToolCall


@dataclass(slots=True)
class ObservationReady:
    result: ToolResult


@dataclass(slots=True)
class RunFinished:
    success: bool
    turns_used: int


@dataclass(slots=True)
class ThinkingDelta:
    """Incremental thinking token during streaming."""

    text: str


@dataclass(slots=True)
class TextDelta:
    """Incremental text token during streaming."""

    text: str


AgentEvent = (
    TurnStarted | AssistantReplied | ToolCalled | ObservationReady
    | RunFinished | ThinkingDelta | TextDelta
)


# Conversation primitives. Providers accept these neutral shapes and
# translate to their wire formats; complete() returns AssistantMessage,
# which doubles as the history entry for the next request.


@dataclass(slots=True)
class UserMessage:
    text: str


@dataclass(slots=True)
class AssistantMessage:
    text: str = ""
    thinking: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0


@dataclass(slots=True)
class ToolResultMessage:
    """All observations from one assistant turn, batched together."""

    results: list[ToolResult] = field(default_factory=list)


ConversationMessage = UserMessage | AssistantMessage | ToolResultMessage


@dataclass(slots=True)
class AgentRunReport:
    """Outcome of one run_agent() invocation."""

    success: bool
    turns_used: int
    max_turns: int
    failure_category: FailureCategory
    failure_summary: str
    turns: list[AgentTurn] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_thinking_tokens: int = 0
    # Final workspace contents (workspace-relative path -> text), captured
    # at termination; size-capped per file. Lets CLIs and JSON consumers
    # see the produced code without touching the workspace itself.
    files: dict[str, str] = field(default_factory=dict)
