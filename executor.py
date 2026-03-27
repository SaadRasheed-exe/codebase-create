from dataclasses import dataclass
from pathlib import Path
import tempfile
import subprocess
import shutil

@dataclass(slots=True)
class ExecutionArtifacts:
    work_dir: Path
    solution_file: Path
    test_file: Path
    junit_file: Path

class TempWorkspace:
    def __init__(self, keep_artifacts: bool = False) -> None:
        self.keep_artifacts = keep_artifacts
        self._tmp_dir = Path(tempfile.mkdtemp(prefix="agent_run_"))

    @property
    def path(self) -> Path:
        return self._tmp_dir

    def write_artifacts(self, implementation: str, tests: str) -> ExecutionArtifacts:
        solution_file = self._tmp_dir / "solution.py"
        test_file = self._tmp_dir / "test_solution.py"
        junit_file = self._tmp_dir / "results.xml"
        solution_file.write_text(implementation, encoding="utf-8")
        test_file.write_text(tests, encoding="utf-8")
        return ExecutionArtifacts(
            work_dir=self._tmp_dir,
            solution_file=solution_file,
            test_file=test_file,
            junit_file=junit_file,
        )

    def cleanup(self) -> None:
        if not self.keep_artifacts:
            shutil.rmtree(self._tmp_dir, ignore_errors=True)


def run_pytest(work_dir: Path, junit_file: Path, timeout_sec: int) -> subprocess.CompletedProcess[str] | None:
    cmd = [
        "python",
        "-m",
        "pytest",
        "-q",
        "--tb=short",
        f"--junitxml={junit_file.name}",
    ]
    try:
        return subprocess.run(
            cmd,
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None
