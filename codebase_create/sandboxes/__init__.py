from codebase_create.config import AgentConfig
from codebase_create.sandboxes.docker_runner import DockerRunner
from codebase_create.sandboxes.subprocess_runner import SubprocessRunner


def get_sandbox_runner(config: AgentConfig):
    if config.sandbox == "docker":
        return DockerRunner(config)
    return SubprocessRunner()
