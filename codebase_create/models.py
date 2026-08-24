from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


FailureCategory = Literal[
    "none",
    "malformed_model_output",
    "syntax_error",
    "runtime_error",
    "test_failure",
    "timeout",
    "infrastructure_error",
    "stuck_loop",
    "max_iterations_reached",
]

@dataclass(slots=True)
class GeneratedArtifacts:
    implementation_code: str
    tests_code: str
    raw_response: str

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

@dataclass(slots=True)
class IterationRecord:
    attempt: int
    artifacts: GeneratedArtifacts
    execution: TestExecutionResult | None = None
    duration_sec: float = 0.0
    temperature: float = 0.0

@dataclass(slots=True)
class FinalReport:
    success: bool
    attempts_used: int
    max_iterations: int
    failure_category: FailureCategory
    failure_summary: str
    records: list[IterationRecord] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Agentic building blocks (new architecture)
#
# These types are provider-neutral: every LLM backend normalizes its native
# tool-calling format (Anthropic tool_use blocks, OpenAI tool_calls JSON
# strings, ...) into ToolCall on the way in, and consumes ToolResult on the
# way back. The orchestrator and UI never see provider-specific shapes.
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


@dataclass(slots=True)
class AgentTurn:
    """One round-trip with the model plus everything it triggered."""

    index: int
    assistant_text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    duration_sec: float = 0.0


@dataclass(slots=True)
class FileInfo:
    """A file inside the agent workspace (path is workspace-relative)."""

    path: str
    bytes: int


# Events emitted by the orchestrator loop. Renderers subscribe to the same
# stream via a callback; matching is done by type so new renderers can be
# added without touching orchestration logic.


@dataclass(slots=True)
class TurnStarted:
    turn_index: int


@dataclass(slots=True)
class AssistantReplied:
    text: str


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


AgentEvent = (
    TurnStarted | AssistantReplied | ToolCalled | ObservationReady | RunFinished
)