from prompts import SYSTEM_PROMPT, build_generation_prompt
from llmbackends import OllamaBackend
from response_parser import parse_model_response
from executor import run_pytest, TempWorkspace


user_request = "Build a Python function that validates email addresses."
model_name = "qwen2.5-coder:3b"

generation_prompt = build_generation_prompt(user_request)
backend = OllamaBackend(model_name=model_name)
response = backend.generate(system_prompt=SYSTEM_PROMPT, user_prompt=generation_prompt)
artifacts = parse_model_response(response)

print("Implementation Code:\n", artifacts.implementation_code)
print("\nTests Code:\n", artifacts.tests_code)


workspace = TempWorkspace(keep_artifacts=False)
try:
    execution_artifacts = workspace.write_artifacts(
        implementation=artifacts.implementation_code,
        tests=artifacts.tests_code,
    )
    test_result = run_pytest(
        work_dir=execution_artifacts.work_dir,
        junit_file=execution_artifacts.junit_file,
        timeout_sec=200,
    )
except Exception as e:
    print("Error during execution:", str(e))
    exit(1)


print("Test Output:\n", test_result.stdout if test_result else "Execution timed out")
print("Test Errors:\n", test_result.stderr if test_result else "Execution timed out")

workspace.cleanup()