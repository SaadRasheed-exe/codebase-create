from ollama import Client

class OllamaBackend:
    def __init__(self, model_name: str):
        self.client = Client()
        self.model_name = model_name

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.message.content