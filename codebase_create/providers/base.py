"""LLM provider layer.

Every backend — mock, Anthropic, or anything speaking the OpenAI
Chat Completions wire format (openai, ollama, nvidia) — normalizes to
the same contract:

- accepts neutral conversation messages (see models.py)
- returns AssistantMessage with parsed ToolCalls and token usage
- raises ProviderError only for configuration/auth/transport problems;
  model-level refusals arrive as ordinary text content

Policy (retries, turn budgets, stop conditions) lives in the
orchestrator, never here.
"""

from abc import ABC, abstractmethod
from typing import Callable

from codebase_create.models import (
    AssistantMessage,
    ConversationMessage,
    ToolSpec,
)

# (kind, delta_text) — kind is "thinking" or "text"
StreamCallback = Callable[[str, str], None]


class ProviderError(RuntimeError):
    """Configuration, authentication, transport, or init failure."""


class Provider(ABC):
    @abstractmethod
    def complete(
        self,
        system_prompt: str,
        messages: list[ConversationMessage],
        tools: list[ToolSpec],
        temperature: float = 0.1,
        on_delta: StreamCallback | None = None,
    ) -> AssistantMessage:
        """One request/response round trip with the model.

        If *on_delta* is provided the provider calls it with incremental
        updates as tokens arrive ("thinking" for reasoning traces, "text"
        for the visible response).  The complete AssistantMessage is still
        returned at the end.  Providers that don't support streaming
        (mock, simple tests) ignore the callback.
        """
