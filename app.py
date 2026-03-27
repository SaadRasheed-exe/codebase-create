from prompts import SYSTEM_PROMPT, build_generation_prompt, build_repair_prompt
from llmbackends import OllamaBackend
from response_parser import parse_model_response, ResponseParseError
from executor import run_pytest, TempWorkspace
from config import AgentConfig
from test_results import parse_test_result
from models import IterationRecord
import time


user_request = "Build a Python function that returns factorial of a number."

config = AgentConfig()
start = time.perf_counter()
generation_prompt = build_generation_prompt(user_request)
backend = OllamaBackend(model_name=config.model)
raw = backend.generate(system_prompt=SYSTEM_PROMPT, user_prompt=generation_prompt)

try:
    artifacts = parse_model_response(raw)
except ResponseParseError as e:
    print(f"Error parsing model response: {e}")
    print(f"Raw model response:\n{raw}")
    exit(1)

print("Implementation Code:\n", artifacts.implementation_code)
print("\nTests Code:\n", artifacts.tests_code)


workspace = TempWorkspace(keep_artifacts=False)

execution_artifacts = workspace.write_artifacts(
    implementation=artifacts.implementation_code,
    tests=artifacts.tests_code,
)
completed = run_pytest(
    work_dir=execution_artifacts.work_dir,
    junit_file=execution_artifacts.junit_file,
    timeout_sec=config.test_timeout_sec,
)

if completed is None:
    execution = parse_test_result("", "", completed.junit_file, timed_out=True)
else:
    execution = parse_test_result(
        completed.stdout,
        completed.stderr,
        execution_artifacts.junit_file,
        timed_out=False
    )

iter_record = IterationRecord(
    artifacts=artifacts,
    execution=execution,
    duration_sec=time.perf_counter() - start,
)

workspace.cleanup()

print("Test Output:\n", completed.stdout if completed else "Execution timed out")
print("Test Errors:\n", completed.stderr if completed else "Execution timed out")

if execution.success:
    print("All tests passed!")

else:

    rebuild_prompt = build_repair_prompt(user_request, iter_record)
    raw = backend.generate(system_prompt=SYSTEM_PROMPT, user_prompt=rebuild_prompt)

    try:
        artifacts = parse_model_response(raw)
    except ResponseParseError as e:
        print(f"Error parsing model response: {e}")
        print(f"Raw model response:\n{raw}")
        exit(1)
    
    workspace = TempWorkspace(keep_artifacts=False)
    
    execution_artifacts = workspace.write_artifacts(
        implementation=artifacts.implementation_code,
        tests=artifacts.tests_code,
    )
    completed = run_pytest(
        work_dir=execution_artifacts.work_dir,
        junit_file=execution_artifacts.junit_file,
        timeout_sec=config.test_timeout_sec,
    )
    
    if completed is None:
        execution = parse_test_result("", "", execution_artifacts.junit_file, timed_out=True)
        print("Rebuild execution timed out")
    else:
        execution = parse_test_result(
            completed.stdout,
            completed.stderr,
            execution_artifacts.junit_file,
            timed_out=False
        )
        print("Rebuild Test Output:\n", completed.stdout)
        print("Rebuild Test Errors:\n", completed.stderr)
    
    if execution.success:
        print("All tests passed after rebuild!")
    else:
        print("Tests still failing after rebuild.")
    
    workspace.cleanup()