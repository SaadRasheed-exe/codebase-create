from pathlib import Path
import tempfile
import subprocess
import shutil

from codebase_create.config import AgentConfig
from codebase_create.models import ExecutionArtifacts, FileInfo
from codebase_create.sandboxes import get_sandbox_runner


MAX_FILE_BYTES = 1_048_576  # defense-in-depth cap on agent-driven writes
_NOISE_FILE_NAMES = {"results.xml"}
_NOISE_DIR_NAMES = {"__pycache__", ".pytest_cache"}


class WorkspacePathError(ValueError):
    """Raised when a relative path attempts to escape the workspace."""


class TempWorkspace:
    """A throwaway directory the agent reads and writes files in.

    All paths are workspace-relative and validated against traversal
    escapes (absolute paths, ``..`` segments, symlinked shortcuts) so a
    confused model can never touch files outside this directory.
    """

    def __init__(self, keep_artifacts: bool = False) -> None:
        self.keep_artifacts = keep_artifacts
        self._tmp_dir = Path(tempfile.mkdtemp(prefix="agent_workspace_"))

    @property
    def path(self) -> Path:
        return self._tmp_dir

    def _resolve(self, rel_path: str) -> Path:
        if not isinstance(rel_path, str) or not rel_path.strip():
            raise WorkspacePathError("Path must be a non-empty string")
        candidate = Path(rel_path)
        if candidate.is_absolute():
            raise WorkspacePathError(f"Absolute paths are not allowed: {rel_path!r}")
        root = self._tmp_dir.resolve()
        resolved = (root / candidate).resolve()
        if not resolved.is_relative_to(root):
            raise WorkspacePathError(f"Path escapes the workspace: {rel_path!r}")
        return resolved

    def write_file(self, rel_path: str, content: str) -> Path:
        target = self._resolve(rel_path)
        payload = content.encode("utf-8")
        if len(payload) > MAX_FILE_BYTES:
            raise ValueError(
                f"Refusing to write {len(payload)} bytes "
                f"(limit is {MAX_FILE_BYTES})"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return target

    def read_file(self, rel_path: str) -> str:
        target = self._resolve(rel_path)
        return target.read_text(encoding="utf-8", errors="replace")

    def list_files(self) -> list[FileInfo]:
        infos: list[FileInfo] = []
        for item in sorted(self._tmp_dir.rglob("*")):
            if not item.is_file():
                continue
            rel = item.relative_to(self._tmp_dir).as_posix()
            if set(rel.split("/")) & _NOISE_DIR_NAMES:
                continue
            if item.name in _NOISE_FILE_NAMES:
                continue
            infos.append(FileInfo(path=rel, bytes=item.stat().st_size))
        return infos

    def write_artifacts(self, implementation: str, tests: str) -> ExecutionArtifacts:
        """Legacy two-file layout, kept until the agentic loop replaces it."""
        solution_file = self.write_file("solution.py", implementation)
        test_file = self.write_file("test_solution.py", tests)
        return ExecutionArtifacts(
            work_dir=self._tmp_dir,
            solution_file=solution_file,
            test_file=test_file,
            junit_file=self._tmp_dir / "results.xml",
        )

    def cleanup(self) -> None:
        if not self.keep_artifacts:
            shutil.rmtree(self._tmp_dir, ignore_errors=True)


def run_pytest(
    work_dir: Path,
    junit_file: Path,
    timeout_sec: int,
    config: AgentConfig,
) -> subprocess.CompletedProcess[str] | None:
    cmd = [
        "python",
        "-m",
        "pytest",
        "-q",
        "--tb=short",
        f"--junitxml={junit_file.name}",
    ]
    runner = get_sandbox_runner(config)
    return runner.run(cmd=cmd, work_dir=work_dir, timeout_sec=timeout_sec)
