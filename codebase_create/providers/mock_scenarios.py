"""Declarative scripts for offline development and testing.

Each scenario is a list of turns the model would plausibly produce.
The MockProvider replays them verbatim, one turn per complete() call,
ignoring conversation history entirely — which is exactly why these
fixtures are useful: they make the loop's behavior deterministic and
document what real providers must handle (tool-call parsing, batched
observations, error results).

Tool names are conformance-checked against TOOL_SPECS at construction,
so renaming a tool breaks loud and early instead of mid-scenario.
"""

from dataclasses import dataclass, field


CORRECT_FACTORIAL = '''\
def factorial(n):
    """Return n! for non-negative integers."""
    if n < 0:
        raise ValueError("factorial is not defined for negative numbers")
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result
'''

RECURSIVE_FACTORIAL_NO_BASE = '''\
def factorial(n):
    return n * factorial(n - 1)
'''

OFF_BY_ONE_FACTORIAL = '''\
def factorial(n):
    if n < 0:
        raise ValueError("factorial is not defined for negative numbers")
    result = 1
    for i in range(2, n):
        result *= i
    return result
'''

FACTORIAL_TESTS = '''\
import pytest

from solution import factorial


def test_base_cases():
    assert factorial(0) == 1
    assert factorial(1) == 1


def test_small_numbers():
    assert factorial(5) == 120


def test_negative_raises():
    with pytest.raises(ValueError):
        factorial(-3)
'''


@dataclass(slots=True)
class ScriptedTurn:
    text: str = ""
    tool_calls: list[tuple[str, dict]] = field(default_factory=list)


SCENARIOS: dict[str, list[ScriptedTurn]] = {
    "happy_path": [
        ScriptedTurn(
            text="I'll implement factorial and write tests.",
            tool_calls=[
                ("write_file", {"path": "solution.py", "content": CORRECT_FACTORIAL}),
                ("write_file", {"path": "test_solution.py", "content": FACTORIAL_TESTS}),
            ],
        ),
        ScriptedTurn(tool_calls=[("run_tests", {})]),
        ScriptedTurn(text="All tests pass. The implementation handles base cases, "
                          "small inputs, and rejects negatives."),
    ],
    "fix_after_failure": [
        ScriptedTurn(
            text="Writing an initial recursive implementation.",
            tool_calls=[
                ("write_file", {"path": "solution.py",
                                 "content": RECURSIVE_FACTORIAL_NO_BASE}),
                ("write_file", {"path": "test_solution.py", "content": FACTORIAL_TESTS}),
            ],
        ),
        ScriptedTurn(tool_calls=[("run_tests", {})]),
        ScriptedTurn(
            text="Tests failed with RecursionError - the recursion has no base case. "
                 "Let me re-read my file and fix it.",
            tool_calls=[
                ("read_file", {"path": "solution.py"}),
                ("write_file", {"path": "solution.py", "content": CORRECT_FACTORIAL}),
            ],
        ),
        ScriptedTurn(tool_calls=[("run_tests", {})]),
        ScriptedTurn(text="Fixed by adding the base case; all tests pass now."),
    ],
    "stuck_loop": [
        ScriptedTurn(
            text="Implementing factorial iteratively.",
            tool_calls=[
                ("write_file", {"path": "solution.py", "content": OFF_BY_ONE_FACTORIAL}),
                ("write_file", {"path": "test_solution.py", "content": FACTORIAL_TESTS}),
            ],
        ),
        ScriptedTurn(tool_calls=[("run_tests", {})]),
        ScriptedTurn(text="Hmm, still failing. Retrying unchanged.", tool_calls=[("run_tests", {})]),
        ScriptedTurn(text="Trying again without changes.", tool_calls=[("run_tests", {})]),
        # Safety valve: if the loop's stuck-detector never fires, this final
        # turn exhausts the script and raises loudly instead of hanging.
        ScriptedTurn(text="One more try.", tool_calls=[("run_tests", {})]),
    ],
    "bad_tool_args": [
        ScriptedTurn(
            text="Writing the implementation.",
            tool_calls=[
                ("write_file", {"path": "solution.py"}),          # missing content
                ("write_file", {"path": "../escape.py", "content": "x = 1"}),  # traversal
            ],
        ),
        ScriptedTurn(text="Both calls were rejected. Let me correct the arguments."),
        ScriptedTurn(
            tool_calls=[
                ("write_file", {"path": "solution.py", "content": CORRECT_FACTORIAL}),
                ("write_file", {"path": "test_solution.py", "content": FACTORIAL_TESTS}),
            ],
        ),
        ScriptedTurn(tool_calls=[("run_tests", {})]),
        ScriptedTurn(text="Recovered after argument errors; all tests pass."),
    ],
}
