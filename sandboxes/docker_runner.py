import subprocess
from io import BytesIO
from pathlib import Path
import re

import docker
from docker.errors import DockerException

from config import AgentConfig


class DockerRunner:
    def __init__(self, config: AgentConfig) -> None:
        self._config = config
        self._client = docker.from_env()
        self._resolved_image: str | None = None

    def run(self, cmd: list[str], work_dir: Path, timeout_sec: int) -> subprocess.CompletedProcess[str] | None:
        image = self._ensure_image()

        nano_cpus = int(self._config.docker_cpus * 1_000_000_000)

        container = self._client.containers.create(
            image=image,
            command=cmd,
            working_dir="/work",
            volumes={str(work_dir.resolve()): {"bind": "/work", "mode": "rw"}},
            network_disabled=self._config.docker_network_disabled,
            mem_limit=self._config.docker_memory_limit,
            nano_cpus=nano_cpus,
            detach=True,
        )

        try:
            container.start()
            try:
                status = container.wait(timeout=timeout_sec)
            except Exception:
                container.kill()
                return None

            exit_code = int(status.get("StatusCode", 1))
            stdout = container.logs(stdout=True, stderr=False).decode("utf-8", errors="replace")
            stderr = container.logs(stdout=False, stderr=True).decode("utf-8", errors="replace")
            return subprocess.CompletedProcess(cmd, exit_code, stdout=stdout, stderr=stderr)
        except DockerException as ex:
            raise RuntimeError(f"Docker execution failed: {ex}") from ex
        finally:
            try:
                container.remove(force=True)
            except DockerException:
                pass

    def _ensure_image(self) -> None:
        if self._resolved_image is not None:
            return self._resolved_image

        base_image = self._config.docker_image
        try:
            self._client.images.get(base_image)
        except DockerException:
            self._client.images.pull(base_image)

        if self._image_has_pytest(base_image):
            self._resolved_image = base_image
            return self._resolved_image

        pytest_image = self._pytest_image_tag(base_image)
        try:
            self._client.images.get(pytest_image)
        except DockerException:
            dockerfile = f"FROM {base_image}\nRUN python -m pip install --no-cache-dir pytest\n"
            self._client.images.build(
                fileobj=BytesIO(dockerfile.encode("utf-8")),
                tag=pytest_image,
                rm=True,
                pull=False,
            )

        self._resolved_image = pytest_image
        return self._resolved_image

    def _image_has_pytest(self, image: str) -> bool:
        try:
            _ = self._client.containers.run(
                image=image,
                command=["python", "-c", "import pytest"],
                network_disabled=False,
                remove=True,
                detach=False,
            )
            return True
        except DockerException:
            return False

    @staticmethod
    def _pytest_image_tag(base_image: str) -> str:
        safe = re.sub(r"[^a-zA-Z0-9_.-]", "-", base_image)
        return f"agent-pytest-{safe}:latest"
