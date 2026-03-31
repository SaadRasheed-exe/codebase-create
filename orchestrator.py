import time
from config import AgentConfig
from executor import TempWorkspace, run_pytest
from llmbackends import OllamaBackend
from models import FinalReport, GeneratedArtifacts, IterationRecord, TestExecutionResult
from prompts import SYSTEM_PROMPT, build_generation_prompt, build_repair_prompt
from response_parser import ResponseParseError, parse_model_response
from test_results import parse_test_result



def run_agent(user_prompt: str, backend: OllamaBackend, config: AgentConfig) -> FinalReport:
    records: list[IterationRecord] = []

    for attempt in range(1, config.max_iterations + 1):
        start = time.perf_counter()

        model_prompt = (
            build_generation_prompt(user_prompt)
            if attempt == 1
            else build_repair_prompt(user_prompt, records[-1])
        )

        try:
            raw = backend.generate(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=model_prompt,
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
            )
            if completed is None:
                execution = parse_test_result("", "", written.junit_file, timed_out=True)
            else:
                execution = parse_test_result(
                    completed.stdout,
                    completed.stderr,
                    written.junit_file,
                    timed_out=False,
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