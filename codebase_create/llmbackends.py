import os
from ollama import Client
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

class OllamaBackend:
    def __init__(self, model_name: str):
        self.client = Client()

        try:
            self.client.models.retrieve(model_name)
        except Exception as e:
            raise ValueError(f"Failed to access model '{model_name}': {e}")

        self.model_name = model_name

    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.1) -> str:
        response = self.client.chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            options={
                "temperature": temperature
            }
        )
        return response.message.content


class OpenAIBackend:
    def __init__(self, model_name: str):
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=os.getenv("nvidia_api_key")
        )

        try:
            self.model_list = self.client.models.list()
            model_list_names = [model.id for model in self.model_list.data]
        except Exception as e:
            raise ValueError(f"Failed to access model list: {e}")
        
        if model_name not in model_list_names:
            raise ValueError(f"Model '{model_name}' not found in available models: {model_list_names}")
        
        self.model_name = model_name

    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.1) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": system_prompt + "\n\n\n" + user_prompt},
            ],
            temperature=temperature
        )
        return response.choices[0].message.content