from dataclasses import dataclass
import os


@dataclass(slots=True)
class AgentConfig:
    model: str = "qwen2.5-coder:3b"
    test_timeout_sec: int = 15
    max_iterations: int = 8
    keep_artifacts: bool = False
    generation_temperature: float = 0.1

    @classmethod
    def from_env(cls) -> "AgentConfig":
        return cls(
            model=os.getenv("AGENT_MODEL", "qwen2.5-coder:3b"),
            test_timeout_sec=int(os.getenv("AGENT_TEST_TIMEOUT", "15")),
            max_iterations=int(os.getenv("AGENT_MAX_ITERATIONS", "8")),
            keep_artifacts=os.getenv("AGENT_KEEP_ARTIFACTS", "false").lower() == "true",
            generation_temperature=float(os.getenv("AGENT_GENERATION_TEMPERATURE", "0.1")),
        )