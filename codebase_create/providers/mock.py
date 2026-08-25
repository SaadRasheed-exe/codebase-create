"""Scripted LLM backend for offline development.

Replays a scenario's turns verbatim, one per complete() call. It never
inspects the conversation history — the orchestrator's messages are
irrelevant to what it returns, which keeps runs fully deterministic.
"""

from codebase_create.models import (
    AssistantMessage,
    ConversationMessage,
    ToolCall,
    ToolSpec,
)
from codebase_create.providers.base import Provider, ProviderError
from codebase_create.providers.mock_scenarios import SCENARIOS, ScriptedTurn
from codebase_create.tools import TOOL_SPECS


class MockProvider(Provider):
    def __init__(self, scenario: str = "happy_path") -> None:
        if scenario not in SCENARIOS:
            valid = ", ".join(sorted(SCENARIOS))
            raise ProviderError(
                f"Unknown mock scenario '{scenario}'. Valid scenarios: {valid}"
            )
        self._scenario_name = scenario
        self._script: list[ScriptedTurn] = SCENARIOS[scenario]
        self._cursor = 0

        known_tools = {spec.name for spec in TOOL_SPECS}
        for turn in self._script:
            for name, _args in turn.tool_calls:
                if name not in known_tools:
                    raise ProviderError(
                        f"Scenario '{scenario}' references unknown tool "
                        f"'{name}'. Update mock_scenarios.py or TOOL_SPECS."
                    )

    def complete(
        self,
        system_prompt: str,
        messages: list[ConversationMessage],
        tools: list[ToolSpec],
        temperature: float = 0.1,
    ) -> AssistantMessage:
        if self._cursor >= len(self._script):
            raise ProviderError(
                f"Mock scenario '{self._scenario_name}' exhausted after "
                f"{len(self._script)} turns; the loop kept calling the model."
            )
        turn = self._script[self._cursor]
        self._cursor += 1
        calls = [
            ToolCall(id=f"mock_{self._cursor}_{i}", name=name, arguments=dict(args))
            for i, (name, args) in enumerate(turn.tool_calls)
        ]
        return AssistantMessage(text=turn.text, thinking=turn.thinking, tool_calls=calls)
