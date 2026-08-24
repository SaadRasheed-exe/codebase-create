import pytest

from codebase_create.config import AgentConfig
from codebase_create.executor import MAX_FILE_BYTES, TempWorkspace
from codebase_create.models import ToolCall
from codebase_create.tools import (
    MAX_FAILURE_MESSAGES,
    MAX_MESSAGE_CHARS,
    MAX_READ_CHARS,
    ToolDispatcher,
    format_test_observation,
)


@pytest.fixture()
def dispatcher():
    ws = TempWorkspace()
    try:
        yield ToolDispatcher(ws, AgentConfig(sandbox="subprocess"))
    finally:
        ws.cleanup()


def call(name, arguments, id="call-1"):
    return ToolCall(id=id, name=name, arguments=arguments)


def execute(dispatcher, name, arguments):
    return dispatcher.execute(call(name, arguments))


def test_write_read_list_roundtrip(dispatcher):
    written = execute(dispatcher, "write_file", {"path": "pkg/mod.py", "content": "x = 42\n"})
    assert not written.is_error
    assert "Wrote 7 bytes to pkg/mod.py" in written.content

    read = execute(dispatcher, "read_file", {"path": "pkg/mod.py"})
    assert read.content == "x = 42\n"

    listing = execute(dispatcher, "list_files", {})
    assert listing.content == "pkg/mod.py (7 bytes)"


def test_list_files_empty_workspace(dispatcher):
    result = execute(dispatcher, "list_files", {})
    assert result.content == "(workspace is empty)"
    assert not result.is_error


def test_unknown_tool_is_error(dispatcher):
    result = dispatcher.execute(call("deploy_to_prod", {}))
    assert result.is_error
    assert "Unknown tool 'deploy_to_prod'" in result.content
    assert "run_tests" in result.content


def test_non_dict_arguments_is_error(dispatcher):
    result = dispatcher.execute(call("list_files", "not a dict"))
    assert result.is_error
    assert "JSON object" in result.content


def test_missing_arguments_is_error(dispatcher):
    result = execute(dispatcher, "write_file", {"path": "a.py"})
    assert result.is_error
    assert "content" in result.content


def test_non_string_argument_is_error(dispatcher):
    result = execute(dispatcher, "write_file", {"path": 123, "content": "x"})
    assert result.is_error
    assert "'path'" in result.content


def test_traversal_attack_returns_error_result(dispatcher):
    result = execute(dispatcher, "write_file", {"path": "../escape.py", "content": "x"})
    assert result.is_error
    assert "escapes the workspace" in result.content


def test_oversize_write_is_error(dispatcher):
    big = "x" * (MAX_FILE_BYTES + 1)
    result = execute(dispatcher, "write_file", {"path": "big.py", "content": big})
    assert result.is_error
    assert "limit" in result.content


def test_read_file_truncates_long_content(dispatcher):
    long_text = "a" * (MAX_READ_CHARS + 500)
    execute(dispatcher, "write_file", {"path": "long.txt", "content": long_text})
    result = execute(dispatcher, "read_file", {"path": "long.txt"})
    assert len(result.content) < len(long_text)
    assert "[truncated 500 chars]" in result.content


def test_run_tests_green_path(dispatcher):
    execute(dispatcher, "write_file", {
        "path": "solution.py",
        "content": "def add(a, b):\n    return a + b\n",
    })
    execute(dispatcher, "write_file", {
        "path": "test_solution.py",
        "content": "from solution import add\n\ndef test_add():\n    assert add(2, 3) == 5\n",
    })
    result = execute(dispatcher, "run_tests", {})
    assert result.content == "PASSED: 1 test(s) passed."
    assert not result.is_error


def test_run_tests_failure_is_observation_not_error(dispatcher):
    execute(dispatcher, "write_file", {"path": "solution.py", "content": "def sub(a, b):\n    return a - b - 1\n"})
    execute(dispatcher, "write_file", {
        "path": "test_solution.py",
        "content": "from solution import sub\n\ndef test_sub():\n    assert sub(5, 2) == 3\n",
    })
    result = execute(dispatcher, "run_tests", {})
    assert not result.is_error
    assert result.content.startswith("FAILED (test_failure):")
    assert "[1]" in result.content


def test_run_tests_reports_no_tests_collected(dispatcher):
    result = execute(dispatcher, "run_tests", {})
    assert result.content.startswith("FAILED (test_failure):")


def test_run_tests_timeout(dispatcher):
    execute(dispatcher, "write_file", {
        "path": "slow_test.py",
        "content": "import time\n\ndef test_slow():\n    time.sleep(30)\n",
    })
    ws_config_ws = dispatcher
    dispatcher._config.test_timeout_sec = 1
    result = execute(dispatcher, "run_tests", {})
    assert result.content == "TIMEOUT: test execution exceeded the time limit."


def test_format_caps_messages_and_length():
    from codebase_create.models import TestExecutionResult

    long_msg = "f" * (MAX_MESSAGE_CHARS * 3)
    result = TestExecutionResult(
        success=False,
        passed=0,
        failed=MAX_FAILURE_MESSAGES + 5,
        errors=0,
        timed_out=False,
        failure_messages=[long_msg] * (MAX_FAILURE_MESSAGES + 5),
        stdout="",
        stderr="",
        category="test_failure",
    )
    text = format_test_observation(result)
    lines = text.splitlines()

    numbered = [line for line in lines if line.startswith("[")]
    assert len(numbered) == MAX_FAILURE_MESSAGES
    # "[N] " prefix + truncated body + "..." suffix
    max_line_len = len(f"[{MAX_FAILURE_MESSAGES}] ") + MAX_MESSAGE_CHARS + 3
    assert all(len(line) <= max_line_len for line in numbered)
    assert all("..." in line for line in numbered)
    assert "more failure message(s) omitted" in lines[-1]
