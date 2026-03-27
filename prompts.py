

SYSTEM_PROMPT = """You are an expert Python developer.
Return only two sections in this exact format:
## IMPLEMENTATION
<python code>
## TESTS
<pytest code>
No markdown fences, no extra prose.
"""


def build_generation_prompt(user_request: str) -> str:
    return f"""Write Python code for this request:

{user_request}

Requirements:
- Produce a complete implementation.
- Produce pytest tests with edge cases.
- Tests should import from solution.py.
"""