from models import IterationRecord


SYSTEM_PROMPT = """You are an expert Python developer.
Important:
Return corrected output in exact given format.
Don't add any explanations.
Do not skip ## IMPLEMENTATION or ## TESTS headings.
No extra pose.

Format:
## IMPLEMENTATION
<python code>
## TESTS
<pytest code>
"""


def build_generation_prompt(user_request: str) -> str:
    return f"""Write Python code for this request:

{user_request}

Requirements:
- Produce a complete implementation.
- Produce pytest tests with edge cases.
- Tests should import from solution.py.
"""

def build_repair_prompt(user_request: str, last: IterationRecord) -> str:
    errors = "\n".join(last.execution.failure_messages[:10]) if last.execution else "Unknown failure"
    return f"""The previous solution failed tests. Fix the code.

Original request:
{user_request}

Previous implementation:
{last.artifacts.implementation_code}

Previous tests:
{last.artifacts.tests_code}

Observed failures:
{errors}

Important:
Return corrected output in exact given format. 
Don't add any explanations.
Do not skip ## IMPLEMENTATION or ## TESTS headings.
No extra pose.

Format:
## IMPLEMENTATION
<python code>
## TESTS
<pytest code>
"""