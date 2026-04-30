import time
from collections import Counter
from config import AgentConfig
from executor import TempWorkspace, run_pytest
from llmbackends import OllamaBackend, OpenAIBackend
from models import FinalReport, GeneratedArtifacts, IterationRecord, TestExecutionResult
from prompts import SYSTEM_PROMPT, build_generation_prompt, build_repair_prompt
from response_parser import ResponseParseError, parse_model_response
from test_results import parse_test_result


def _dominant_error(execution: TestExecutionResult) -> str:
    if execution.failure_messages:
        return execution.failure_messages[0][:200]
    return execution.category


def _compute_temperature(attempt: int, records: list[IterationRecord], config: AgentConfig) -> float:
    base = config.generation_temperature
    ramp_per_attempt = 0.06
    temp = base + ramp_per_attempt * max(attempt - 1, 0)

    if records:
        last = records[-1].execution
        if last is not None:
            if last.category in ("malformed_model_output", "syntax_error"):
                temp -= 0.08
            elif last.category in ("test_failure", "runtime_error"):
                temp += 0.05
            elif last.category == "timeout":
                temp += 0.03

    # if last 2 attempts failed with same category + first message, boost diversity.
    if len(records) >= 2:
        def fingerprint(r: IterationRecord) -> tuple[str, str]:
            ex = r.execution
            if ex is None:
                return ("none", "")
            first_msg = ex.failure_messages[0] if ex.failure_messages else ""
            return (ex.category, first_msg[:160])

        if fingerprint(records[-1]) == fingerprint(records[-2]):
            temp += 0.10

    return max(0.08, min(temp, 0.50))


def run_agent(user_prompt: str, backend: OllamaBackend | OpenAIBackend, config: AgentConfig) -> FinalReport:
    records: list[IterationRecord] = []
    repeated_error_tracker: Counter[str] = Counter()

    for attempt in range(1, config.max_iterations + 1):
        start = time.perf_counter()

        model_prompt = (
            build_generation_prompt(user_prompt)
            if attempt == 1
            else build_repair_prompt(user_prompt, records[-1])
        )

        attempt_temperature = _compute_temperature(attempt, records, config)
        try:
            raw = backend.generate(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=model_prompt,
                temperature=attempt_temperature,
            )
            artifacts = parse_model_response(raw)
        except ResponseParseError as ex:
            artifacts = GeneratedArtifacts("", "", str(ex))
            execution = TestExecutionResult(
                success=False,
                passed=0,
                failed=0,
                errors=1,
                timed_out=False,
                failure_messages=[f"Malformed model output: {ex}"],
                stdout="",
                stderr="",
                category="malformed_model_output",
            )
            record = IterationRecord(
                attempt=attempt,
                artifacts=artifacts,
                execution=execution,
                duration_sec=time.perf_counter() - start,
                temperature=attempt_temperature,
            )
            records.append(record)
            continue
    
        workspace = TempWorkspace(keep_artifacts=config.keep_artifacts)
        try:
            written = workspace.write_artifacts(
                implementation=artifacts.implementation_code,
                tests=artifacts.tests_code,
            )

            completed = run_pytest(
                work_dir=written.work_dir,
                junit_file=written.junit_file,
                timeout_sec=config.test_timeout_sec,
                config=config,
            )
            if completed is None:
                execution = parse_test_result("", "", written.junit_file, timed_out=True)
            else:
                execution = parse_test_result(
                    completed.stdout,
                    completed.stderr,
                    written.junit_file,
                    timed_out=False,
                    exit_code=completed.returncode,
                )
        except Exception as ex:  # pragma: no cover
            execution = TestExecutionResult(
                success=False,
                passed=0,
                failed=0,
                errors=1,
                timed_out=False,
                failure_messages=[f"Infrastructure error: {ex}"],
                stdout="",
                stderr="",
                category="infrastructure_error",
            )
        finally:
            workspace.cleanup()
        
        record = IterationRecord(
            attempt=attempt,
            artifacts=artifacts,
            execution=execution,
            duration_sec=time.perf_counter() - start,
            temperature=attempt_temperature,
        )
        records.append(record)
    
        if execution.success:
            return FinalReport(
                success=True,
                attempts_used=attempt,
                max_iterations=config.max_iterations,
                failure_category="none",
                failure_summary="All tests passed",
                records=records,
            )
        
        key = _dominant_error(execution)
        repeated_error_tracker[key] += 1
        if repeated_error_tracker[key] >= 3:
            return FinalReport(
                success=False,
                attempts_used=attempt,
                max_iterations=config.max_iterations,
                failure_category="stuck_loop",
                failure_summary=f"Repeated failure pattern detected: {key}",
                records=records,
            )
    
    last = records[-1].execution if records else None
    return FinalReport(
        success=False,
        attempts_used=len(records),
        max_iterations=config.max_iterations,
        failure_category=(last.category if last else "max_iterations_reached"),
        failure_summary=(
            last.failure_messages[0] if last and last.failure_messages else "Reached max iterations"
        ),
        records=records,
    )