from ollama import Client

client = Client()

user_request = "Write a function in Python that calculates the factorial of a number."

SYSTEM_PROMPT = """You are an expert Python developer.
Return only two sections in this exact format:
## IMPLEMENTATION
<python code>
## TESTS
<pytest code>
No markdown fences, no extra prose.
"""

prompt = f"""Write Python code for this request:

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
        {"role": "user", "content": prompt},
    ],
)

print(response.message.content)