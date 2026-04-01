import re
import xml.etree.ElementTree as ET
from pathlib import Path
from models import TestExecutionResult


__test__ = False
SUMMARY_RE = re.compile(r"(\d+) failed|((\d+) passed)")


def _extract_messages_from_xml(junit_file: Path) -> tuple[int, int, int, list[str]]:
    if not junit_file.exists():
        return 0, 0, 0, []

    tree = ET.parse(junit_file)
    root = tree.getroot()

    tests = 0
    failures = 0
    errors = 0
    messages: list[str] = []

    for case in root.iter("testcase"):
        tests += 1
        failure = case.find("failure")
        error = case.find("error")
        if failure is not None and failure.text:
            messages.append(failure.text.strip())
            failures += 1
        if error is not None and error.text:
            messages.append(error.text.strip())
            errors += 1

    passed = max(tests - failures - errors, 0)
    return passed, failures, errors, messages


def parse_test_result(
    stdout: str,
    stderr: str,
    junit_file: Path,
    timed_out: bool,
    exit_code: int | None = None,
) -> TestExecutionResult:
    if timed_out:
        return TestExecutionResult(
            success=False,
            passed=0,
            failed=0,
            errors=1,
            timed_out=True,
            failure_messages=["Execution timed out"],
            stdout=stdout,
            stderr=stderr,
            category="timeout",
        )

    passed, failed, errors, messages = _extract_messages_from_xml(junit_file)

    if not messages:
        text = (stdout + "\n" + stderr).strip()
        if text:
            messages.append(text[:2000])

    text_blob = f"{stdout}\n{stderr}"
    lower_text = text_blob.lower()
    total_tests = passed + failed + errors

    category = "none"
    if failed > 0:
        category = "test_failure"
    elif errors > 0:
        if "SyntaxError" in stderr or "SyntaxError" in stdout:
            category = "syntax_error"
        else:
            category = "runtime_error"
    elif "Traceback" in text_blob:
        category = "runtime_error"
    elif "SyntaxError" in text_blob:
        category = "syntax_error"
    elif "NameError" in text_blob:
        category = "runtime_error"

    if exit_code is not None and exit_code != 0 and failed == 0 and errors == 0:
        category = "infrastructure_error"
        errors = 1
        if not messages:
            messages.append("Pytest exited with a non-zero status before producing structured results.")

    if total_tests == 0:
        if "no tests ran" in lower_text or "collected 0 items" in lower_text:
            category = "test_failure"
            failed = max(failed, 1)
            if not messages:
                messages.append("No tests were collected or executed.")
        elif not timed_out:
            category = "infrastructure_error"
            errors = max(errors, 1)
            if not messages:
                messages.append("No JUnit test results were produced.")

    return TestExecutionResult(
        success=(failed == 0 and errors == 0 and total_tests > 0 and category == "none"),
        passed=passed,
        failed=failed,
        errors=errors,
        timed_out=False,
        failure_messages=messages,
        stdout=stdout,
        stderr=stderr,
        category=category,
    )
