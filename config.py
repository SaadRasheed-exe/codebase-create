from dataclasses import dataclass
import os


@dataclass(slots=True)
class AgentConfig:
    model: str = "qwen2.5-coder:3b"
    test_timeout_sec: int = 15

    @classmethod
    def from_env(cls) -> "AgentConfig":
        return cls(
            model=os.getenv("AGENT_MODEL", "qwen2.5-coder:3b"),
            test_timeout_sec=int(os.getenv("AGENT_TEST_TIMEOUT", "15")),
        )