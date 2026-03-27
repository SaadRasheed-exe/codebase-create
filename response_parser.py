from models import GeneratedArtifacts


class ResponseParseError(ValueError):
    pass


def parse_model_response(raw: str) -> GeneratedArtifacts:
    impl_tag = "## IMPLEMENTATION"
    tests_tag = "## TESTS"

    impl_idx = raw.find(impl_tag)
    tests_idx = raw.find(tests_tag)

    if impl_idx == -1 or tests_idx == -1 or tests_idx < impl_idx:
        raise ResponseParseError("Model output missing required sections")

    impl_start = impl_idx + len(impl_tag)
    impl_code = raw[impl_start:tests_idx].strip()
    tests_code = raw[tests_idx + len(tests_tag):].strip()
    impl_code = impl_code.replace("```python", "").replace("```", "").strip()
    tests_code = tests_code.replace("```python", "").replace("```", "").strip()

    if not impl_code or not tests_code:
        raise ResponseParseError("Implementation or tests section is empty")

    return GeneratedArtifacts(
        implementation_code=impl_code,
        tests_code=tests_code,
        raw_response=raw,
    )
