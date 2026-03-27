from ollama import Client
from dataclasses import dataclass
from pathlib import Path
import tempfile
import subprocess

client = Client()

user_request = "Build a Python function that validates email addresses."

SYSTEM_PROMPT = """You are an expert Python developer.
Return only two sections in this exact format:
## IMPLEMENTATION
<python code>
## TESTS
<pytest code>
No markdown fences, no extra prose.
"""

generation_prompt = f"""Write Python code for this request:

{user_request}

Requirements:
- Produce a complete implementation.
- Produce pytest tests with edge cases.
- Tests should import from solution.py.
"""



response = client.chat(
    model="qwen2.5-coder:3b",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": generation_prompt},
    ],
)

class ResponseParseError(ValueError):
    pass

raw = response.message.content
impl_tag = "## IMPLEMENTATION"
tests_tag = "## TESTS"

impl_idx = raw.find(impl_tag)
tests_idx = raw.find(tests_tag)

if impl_idx == -1 or tests_idx == -1 or tests_idx < impl_idx:
    raise ResponseParseError("Model output missing required sections")

impl_start = impl_idx + len(impl_tag)
impl_code = raw[impl_start:tests_idx].strip()
tests_code = raw[tests_idx + len(tests_tag):].strip()
impl_code = impl_code.strip("`").strip("python")
tests_code = tests_code.strip("`").strip("python")

if not impl_code or not tests_code:
    raise ResponseParseError("Implementation or tests section is empty")


@dataclass(slots=True)
class GeneratedArtifacts:
    implementation_code: str
    tests_code: str
    raw_response: str

artifacts = GeneratedArtifacts(
    implementation_code=impl_code,
    tests_code=tests_code,
    raw_response=raw,
)

print("Implementation Code:\n", artifacts.implementation_code)
print("\nTests Code:\n", artifacts.tests_code)


temp_dir = tempfile.mkdtemp()
impl_file = Path(temp_dir) / "solution.py"
test_file = Path(temp_dir) / "test_solution.py"
junit_file = Path(temp_dir) / "results.xml"

impl_file.write_text(artifacts.implementation_code, encoding="utf-8")
test_file.write_text(artifacts.tests_code, encoding="utf-8")


@dataclass(slots=True)
class ExecutionArtifacts:
    work_dir: Path
    solution_file: Path
    test_file: Path
    junit_file: Path

execution_artifacts = ExecutionArtifacts(
    work_dir=Path(temp_dir),
    solution_file=impl_file,
    test_file=test_file,
    junit_file=junit_file,
)

cmd = [
    "python",
    "-m",
    "pytest",
    "-q",
    "--tb=short",
    f"--junitxml={execution_artifacts.junit_file.name}",
]
try:
    res = subprocess.run(
        cmd,
        cwd=execution_artifacts.work_dir,
        capture_output=True,
        text=True,
        timeout=200,
        check=False,
    )
except subprocess.TimeoutExpired:
    res = None


print("Test Output:\n", res.stdout if res else "Execution timed out")
print("Test Errors:\n", res.stderr if res else "Execution timed out")

