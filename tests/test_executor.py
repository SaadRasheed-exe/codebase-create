import os

import pytest

from codebase_create.executor import (
    MAX_FILE_BYTES,
    TempWorkspace,
    WorkspacePathError,
)


def test_write_and_read_roundtrip_nested_unicode():
    ws = TempWorkspace()
    try:
        ws.write_file("pkg/utils.py", "VALUE = 'caf\u00e9'\n")
        assert ws.read_file("pkg/utils.py") == "VALUE = 'caf\u00e9'\n"
    finally:
        ws.cleanup()


def test_write_creates_parent_dirs():
    ws = TempWorkspace()
    try:
        target = ws.write_file("a/b/c.py", "x = 1\n")
        assert target.exists()
        assert (ws.path / "a" / "b" / "c.py").exists()
    finally:
        ws.cleanup()


def test_overwrite_existing_file():
    ws = TempWorkspace()
    try:
        ws.write_file("mod.py", "first")
        ws.write_file("mod.py", "second")
        assert ws.read_file("mod.py") == "second"
    finally:
        ws.cleanup()


@pytest.mark.parametrize("bad", ["", "   ", "/etc/passwd", "../escape.txt"])
def test_reject_invalid_paths(bad):
    ws = TempWorkspace()
    try:
        with pytest.raises(WorkspacePathError):
            ws.write_file(bad, "content")
        with pytest.raises(WorkspacePathError):
            ws.read_file(bad)
    finally:
        ws.cleanup()


def test_reject_deep_parent_escape():
    ws = TempWorkspace()
    try:
        with pytest.raises(WorkspacePathError):
            ws.write_file("a/b/../../../escape.txt", "nope")
    finally:
        ws.cleanup()


def test_reject_symlink_escape(tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")

    ws = TempWorkspace()
    try:
        (ws.path / "link").symlink_to(outside)
        with pytest.raises(WorkspacePathError):
            ws.read_file("link")
        with pytest.raises(WorkspacePathError):
            ws.write_file("link", "tampered")
        assert outside.read_text() == "secret"
    finally:
        ws.cleanup()


def test_write_rejects_oversized_content():
    ws = TempWorkspace()
    try:
        big = "x" * (MAX_FILE_BYTES + 1)
        with pytest.raises(ValueError, match="limit"):
            ws.write_file("big.py", big)
    finally:
        ws.cleanup()


def test_list_files_sorted_relative_and_filtered():
    ws = TempWorkspace()
    try:
        ws.write_file("zeta.py", "1")
        ws.write_file("pkg/alpha.py", "2")
        ws.write_file("results.xml", "<xml/>")  # junit noise
        noise_dir = ws.path / "__pycache__"
        noise_dir.mkdir()
        (noise_dir / "cached.pyc").write_bytes(b"\x00\x01")

        listed = ws.list_files()

        assert [f.path for f in listed] == ["pkg/alpha.py", "zeta.py"]
        assert all(f.bytes > 0 for f in listed)
    finally:
        ws.cleanup()


def test_cleanup_removes_workspace():
    ws = TempWorkspace()
    path = ws.path
    ws.write_file("keep.txt", "data")
    ws.cleanup()
    assert not path.exists()


def test_keep_artifacts_preserves_workspace():
    ws = TempWorkspace(keep_artifacts=True)
    path = ws.path
    ws.write_file("keep.txt", "data")
    ws.cleanup()
    assert path.exists()
    assert (path / "keep.txt").read_text(encoding="utf-8") == "data"


def test_write_artifacts_legacy_shim():
    ws = TempWorkspace()
    try:
        artifacts = ws.write_artifacts("impl = 1", "def test_x():\n    pass\n")
        assert artifacts.work_dir == ws.path
        assert artifacts.solution_file.read_text(encoding="utf-8") == "impl = 1"
        assert artifacts.test_file.name == "test_solution.py"
        assert artifacts.junit_file.name == "results.xml"
    finally:
        ws.cleanup()


def test_files_are_utf8_on_disk():
    ws = TempWorkspace()
    try:
        ws.write_file("u.py", "# \u2713 unicode marker\n")
        raw = (ws.path / "u.py").read_bytes()
        raw.decode("utf-8")
        os.utime(ws.path / "u.py")  # touch to confirm file exists on disk
    finally:
        ws.cleanup()
