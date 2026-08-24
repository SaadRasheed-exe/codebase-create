from dataclasses import dataclass
import os


@dataclass(slots=True)
class AgentConfig:
    backend: str = "openai"
    model: str = "google/gemma-2-2b-it"
    test_timeout_sec: int = 15
    max_iterations: int = 8
    keep_artifacts: bool = False
    generation_temperature: float = 0.1
    sandbox: str = "docker"
    docker_image: str = "python:3.11-slim"
    docker_network_disabled: bool = True
    docker_memory_limit: str = "512m"
    docker_cpus: float = 1.0

    @classmethod
    def from_env(cls) -> "AgentConfig":
        return cls(
            backend=os.getenv("AGENT_BACKEND", "openai").lower(),
            model=os.getenv("AGENT_MODEL", "google/gemma-2-2b-it"),
            test_timeout_sec=int(os.getenv("AGENT_TEST_TIMEOUT", "15")),
            max_iterations=int(os.getenv("AGENT_MAX_ITERATIONS", "8")),
            keep_artifacts=os.getenv("AGENT_KEEP_ARTIFACTS", "false").lower() == "true",
            generation_temperature=float(os.getenv("AGENT_GENERATION_TEMPERATURE", "0.1")),
            sandbox=os.getenv("AGENT_SANDBOX", "docker").lower(),
            docker_image=os.getenv("AGENT_DOCKER_IMAGE", "python:3.11-slim"),
            docker_network_disabled=os.getenv("AGENT_DOCKER_NETWORK_DISABLED", "true").lower() == "true",
            docker_memory_limit=os.getenv("AGENT_DOCKER_MEMORY", "512m"),
            docker_cpus=float(os.getenv("AGENT_DOCKER_CPUS", "1.0")),
        )