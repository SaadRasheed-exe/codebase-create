"""Adapter for anything speaking the OpenAI Chat Completions wire format.

One implementation serves three backends via configuration:

- ``openai``  : default endpoints, OPENAI_API_KEY
- ``nvidia``  : https://integrate.api.nvidia.com/v1 (legacy lowercase
                ``nvidia_api_key`` env var, model-list validation)
- ``ollama``  : http://localhost:11434/v1 (model-existence validation)

Conversion helpers are module-level and side-effect-free so wire-format
behavior can be tested without network access.
"""

import json

from openai import OpenAI

from codebase_create.models import (
    AssistantMessage,
    ConversationMessage,
    ToolSpec,
    ToolCall,
    UserMessage,
    ToolResultMessage,
)
from codebase_create.providers.base import Provider, ProviderError


def to_openai_messages(
    system_prompt: str,
    messages: list[ConversationMessage],
) -> list[dict]:
    """Translate neutral conversation history into OpenAI message dicts."""
    out: list[dict] = [{"role": "system", "content": system_prompt}]
    for message in messages:
        if isinstance(message, UserMessage):
            out.append({"role": "user", "content": message.text})
        elif isinstance(message, AssistantMessage):
            if not message.tool_calls:
                out.append({"role": "assistant", "content": message.text})
                continue
            out.append({
                "role": "assistant",
                # strict endpoints prefer null over "" alongside tool_calls
                "content": message.text or None,
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(call.arguments),
                        },
                    }
                    for call in message.tool_calls
                ],
            })
        elif isinstance(message, ToolResultMessage):
            for result in message.results:
                content = f"[error] {result.content}" if result.is_error else result.content
                out.append({
                    "role": "tool",
                    "tool_call_id": result.call_id,
                    "content": content,
                })
        else:
            raise ProviderError(f"Unhandled message type: {type(message).__name__}")
    return out


def openai_tools_from_specs(specs: list[ToolSpec]) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.parameters,
            },
        }
        for spec in specs
    ]


def parse_openai_response(response) -> AssistantMessage:
    choice = response.choices[0]
    raw_message = choice.message
    calls = []
    for tc in getattr(raw_message, "tool_calls", None) or []:
        try:
            arguments = json.loads(tc.function.arguments or "{}")
        except json.JSONDecodeError as ex:
            raise ProviderError(
                f"Model emitted malformed JSON arguments for "
                f"'{tc.function.name}': {ex}"
            ) from ex
        calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=arguments))

    usage = getattr(response, "usage", None)
    return AssistantMessage(
        text=getattr(raw_message, "content", None) or "",
        tool_calls=calls,
        input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        output_tokens=getattr(usage, "completion_tokens", 0) or 0,
    )


def _validate_nvidia_model(client: OpenAI, model_name: str) -> None:
    """Ported from legacy llmbackends.OpenAIBackend init."""
    try:
        available = [m.id for m in client.models.list().data]
    except Exception as ex:
        raise ProviderError(f"Failed to access NVIDIA model list: {ex}") from ex
    if model_name not in available:
        raise ProviderError(
            f"Model '{model_name}' not found on NVIDIA endpoint. "
            f"Available models include: {available[:10]}..."
        )


def _validate_ollama_model(client: OpenAI, model_name: str) -> None:
    try:
        client.models.retrieve(model_name)
    except Exception as ex:
        raise ProviderError(
            f"Ollama model '{model_name}' unavailable. Is the server "
            f"running and the model pulled? ({ex})"
        ) from ex


class OpenAICompatProvider(Provider):
    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        validate_model=None,
    ) -> None:
        self._model_name = model_name
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        if validate_model is not None:
            validate_model(self._client, model_name)

    def complete(
        self,
        system_prompt: str,
        messages: list[ConversationMessage],
        tools: list[ToolSpec],
        temperature: float = 0.1,
    ) -> AssistantMessage:
        kwargs: dict = {
            "model": self._model_name,
            "messages": to_openai_messages(system_prompt, messages),
            "temperature": temperature,
        }
        if tools:  # several strict endpoints reject an empty tools array
            kwargs["tools"] = openai_tools_from_specs(tools)

        try:
            response = self._client.chat.completions.create(**kwargs)
        except ProviderError:
            raise
        except Exception as ex:
            raise ProviderError(f"Chat completion request failed: {ex}") from ex
        return parse_openai_response(response)
