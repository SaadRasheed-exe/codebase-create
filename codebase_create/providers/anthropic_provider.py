"""Native Anthropic adapter.

Wire-format differences from the OpenAI-compatible world:

- the system prompt is a top-level parameter, not a message
- tool definitions use ``input_schema`` instead of ``parameters``
- tool results travel as ``tool_result`` blocks inside a *user* message
  immediately after the assistant turn that requested them; all results
  from one turn are batched into that single user message
- ``tool_result`` supports a native ``is_error`` flag (no prefix hacks)
- max_tokens is required by the API
"""

from codebase_create.models import (
    AssistantMessage,
    ConversationMessage,
    ToolSpec,
    ToolCall,
    UserMessage,
    ToolResultMessage,
)
from codebase_create.providers.base import Provider, ProviderError, StreamCallback


def to_anthropic_messages(messages: list[ConversationMessage]) -> list[dict]:
    out: list[dict] = []
    for message in messages:
        if isinstance(message, UserMessage):
            out.append({
                "role": "user",
                "content": [{"type": "text", "text": message.text}],
            })
        elif isinstance(message, AssistantMessage):
            blocks: list[dict] = []
            if message.text:
                blocks.append({"type": "text", "text": message.text})
            for call in message.tool_calls:
                blocks.append({
                    "type": "tool_use",
                    "id": call.id,
                    "name": call.name,
                    "input": call.arguments,
                })
            if blocks:
                out.append({"role": "assistant", "content": blocks})
        elif isinstance(message, ToolResultMessage):
            out.append({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": result.call_id,
                        "content": result.content,
                        "is_error": result.is_error,
                    }
                    for result in message.results
                ],
            })
        else:
            raise ProviderError(f"Unhandled message type: {type(message).__name__}")
    return out


def anthropic_tools_from_specs(specs: list[ToolSpec]) -> list[dict]:
    return [
        {
            "name": spec.name,
            "description": spec.description,
            "input_schema": spec.parameters,
        }
        for spec in specs
    ]


def parse_anthropic_response(response) -> AssistantMessage:
    text_parts: list[str] = []
    thinking_parts: list[str] = []
    calls: list[ToolCall] = []
    for block in response.content:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "tool_use":
            calls.append(ToolCall(id=block.id, name=block.name, arguments=dict(block.input)))
        elif block.type == "thinking":
            thinking_parts.append(block.thinking)
        elif block.type == "redacted_thinking":
            pass  # skip redacted blocks silently
        else:
            raise ProviderError(f"Unexpected response block type: {block.type}")

    usage = getattr(response, "usage", None)
    output_tokens = getattr(usage, "output_tokens", 0) or 0
    thinking_tokens = 0
    details = getattr(usage, "output_tokens_details", None)
    if details:
        thinking_tokens = getattr(details, "thinking_tokens", 0) or 0
        output_tokens -= thinking_tokens

    return AssistantMessage(
        text="\n".join(text_parts),
        thinking="\n".join(thinking_parts),
        tool_calls=calls,
        input_tokens=getattr(usage, "input_tokens", 0) or 0,
        output_tokens=output_tokens,
        thinking_tokens=thinking_tokens,
    )


class AnthropicProvider(Provider):
    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        max_tokens: int = 4096,
        enable_thinking: bool = False,
        thinking_budget_tokens: int = 10000,
    ) -> None:
        import anthropic  # deferred so offline/mock usage needs no SDK

        self._model_name = model_name
        self._max_tokens = max_tokens
        self._enable_thinking = enable_thinking
        self._thinking_budget_tokens = thinking_budget_tokens
        try:
            self._client = anthropic.Anthropic(api_key=api_key)
        except Exception as ex:
            raise ProviderError(f"Failed to create Anthropic client: {ex}") from ex

    def complete(
        self,
        system_prompt: str,
        messages: list[ConversationMessage],
        tools: list[ToolSpec],
        temperature: float = 0.1,
        on_delta: StreamCallback | None = None,
    ) -> AssistantMessage:
        kwargs: dict = {
            "model": self._model_name,
            "max_tokens": self._max_tokens,
            "system": system_prompt,
            "messages": to_anthropic_messages(messages),
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = anthropic_tools_from_specs(tools)
        if self._enable_thinking:
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": self._thinking_budget_tokens,
            }
            # thinking needs headroom above the budget
            kwargs["max_tokens"] = max(self._max_tokens, self._thinking_budget_tokens + 1024)

        # --- streaming path ---
        if on_delta is not None:
            try:
                with self._client.messages.stream(**kwargs) as stream:
                    for event in stream:
                        if event.type == "content_block_delta":
                            delta = event.delta
                            if delta.type == "thinking_delta":
                                if not on_delta("thinking", delta.thinking):
                                    break
                            elif delta.type == "text_delta":
                                if not on_delta("text", delta.text):
                                    break
                    response = stream.get_final_message()
            except ProviderError:
                raise
            except Exception as ex:
                raise ProviderError(f"Anthropic stream failed: {ex}") from ex

            if getattr(response, "stop_reason", None) == "max_tokens":
                raise ProviderError(
                    "Anthropic response hit max_tokens before completing; "
                    "raise AGENT_MAX_TOKENS."
                )
            return parse_anthropic_response(response)

        # --- non-streaming path (unchanged) ---
        try:
            response = self._client.messages.create(**kwargs)
        except ProviderError:
            raise
        except Exception as ex:
            raise ProviderError(f"Anthropic request failed: {ex}") from ex

        if getattr(response, "stop_reason", None) == "max_tokens":
            raise ProviderError(
                "Anthropic response hit max_tokens before completing; "
                "raise AGENT_MAX_TOKENS."
            )
        return parse_anthropic_response(response)
