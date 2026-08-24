import pytest

from codebase_create.response_parser import ResponseParseError, parse_model_response


VALID = """Some preamble text.

## IMPLEMENTATION
```python
def add(a, b):
    return a + b
```

## TESTS
```python
def test_add():
    assert add(1, 2) == 3
```
"""


def test_parses_valid_output_and_strips_fences():
    artifacts = parse_model_response(VALID)
    assert artifacts.implementation_code == "def add(a, b):\n    return a + b"
    assert "def test_add" in artifacts.tests_code
    assert artifacts.raw_response == VALID


def test_missing_implementation_marker_raises():
    raw = "## TESTS\ndef test_x():\n    pass\n"
    with pytest.raises(ResponseParseError):
        parse_model_response(raw)


def test_missing_tests_marker_raises():
    raw = "## IMPLEMENTATION\nx = 1\n"
    with pytest.raises(ResponseParseError):
        parse_model_response(raw)


def test_reversed_marker_order_raises():
    raw = "## TESTS\nfirst\n## IMPLEMENTATION\nsecond\n"
    with pytest.raises(ResponseParseError):
        parse_model_response(raw)


def test_empty_sections_raise():
    raw = "## IMPLEMENTATION\n\n## TESTS\n\n"
    with pytest.raises(ResponseParseError):
        parse_model_response(raw)
