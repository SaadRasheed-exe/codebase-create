from config import AgentConfig
from sandboxes.docker_runner import DockerRunner
from sandboxes.subprocess_runner import SubprocessRunner


def get_sandbox_runner(config: AgentConfig):
    if config.sandbox == "docker":
        return DockerRunner(config)
    return SubprocessRunner()
