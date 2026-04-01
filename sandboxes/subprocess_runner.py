import subprocess
from pathlib import Path


class SubprocessRunner:
    def run(self, cmd: list[str], work_dir: Path, timeout_sec: int) -> subprocess.CompletedProcess[str] | None:
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
