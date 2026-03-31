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

@dataclass(slots=True)
class FinalReport:
    success: bool
    attempts_used: int
    max_iterations: int
    failure_category: FailureCategory
    failure_summary: str
    records: list[IterationRecord] = field(default_factory=list)