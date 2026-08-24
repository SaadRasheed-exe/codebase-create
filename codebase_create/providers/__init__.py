"""Provider factory: maps config.backend to a concrete Provider."""

import os

from codebase_create.config import AgentConfig
from codebase_create.providers.anthropic_provider import AnthropicProvider
from codebase_create.providers.base import Provider, ProviderError
from codebase_create.providers.mock import MockProvider
from codebase_create.providers.openai_compat import (
    OpenAICompatProvider,
    _validate_nvidia_model,
    _validate_ollama_model,
)


NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
OLLAMA_BASE_URL = "http://localhost:11434/v1"

BACKENDS = ("mock", "anthropic", "openai", "ollama", "nvidia")


def build_provider(config: AgentConfig) -> Provider:
    backend = config.backend.lower()

    if backend == "mock":
        return MockProvider(config.mock_scenario)

    if backend == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ProviderError(
                "Backend 'anthropic' requires ANTHROPIC_API_KEY "
                "(set it in your environment or .env)."
            )
        return AnthropicProvider(
            config.model, api_key=api_key, max_tokens=config.max_tokens
        )

    if backend == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ProviderError(
                "Backend 'openai' requires OPENAI_API_KEY "
                "(set it in your environment or .env)."
            )
        return OpenAICompatProvider(
            config.model,
            base_url=os.getenv("OPENAI_BASE_URL") or None,
            api_key=api_key,
        )

    if backend == "ollama":
        return OpenAICompatProvider(
            config.model,
            base_url=os.getenv("OLLAMA_BASE_URL") or OLLAMA_BASE_URL,
            api_key="ollama",  # the local server does not check keys
            validate_model=_validate_ollama_model,
        )

    if backend == "nvidia":
        # legacy lowercase var name kept for compatibility with existing .env files
        api_key = os.getenv("nvidia_api_key")
        if not api_key:
            raise ProviderError(
                "Backend 'nvidia' requires nvidia_api_key "
                "(set it in your environment or .env)."
            )
        return OpenAICompatProvider(
            config.model,
            base_url=NVIDIA_BASE_URL,
            api_key=api_key,
            validate_model=_validate_nvidia_model,
        )

    raise ValueError(
        f"Unsupported backend '{config.backend}'. Valid backends: {', '.join(BACKENDS)}"
    )
