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

from codebase_create.models import (
    AssistantMessage,
    ConversationMessage,
    ToolSpec,
)


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
    ) -> AssistantMessage:
        """One request/response round trip with the model."""
