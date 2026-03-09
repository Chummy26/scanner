from pathlib import Path

import pytest

import src.server as server_module


class _FakeProc:
    def __init__(self, pid: int, cmdline: list[str], name: str = "python.exe"):
        self.info = {"pid": pid, "cmdline": cmdline, "name": name}


def test_looks_like_server_process_matches_module_and_script_forms():
    assert server_module._looks_like_server_process(["python", "-m", "src.server", "--port", "8000"]) is True
    assert server_module._looks_like_server_process(["python", "backend/src/server.py"]) is True
    assert server_module._looks_like_server_process(["python", "something_else.py"]) is False


def test_looks_like_server_process_rejects_wrapper_processes():
    """cmd.exe, bash.exe, and timeout.exe wrappers must NOT match."""
    assert server_module._looks_like_server_process(
        ["cmd", "/c", "start", "/b", ".venv\\Scripts\\python.exe", "-u", "src/server.py"]
    ) is False
    assert server_module._looks_like_server_process(
        ["bash", "-c", ".venv/Scripts/python.exe -u src/server.py"]
    ) is False
    assert server_module._looks_like_server_process(
        ["timeout", "12", ".venv/Scripts/python.exe", "-u", "src/server.py"]
    ) is False
    assert server_module._looks_like_server_process(
        ["C:\\Program Files\\Git\\usr\\bin\\timeout.exe", "15", ".venv/Scripts/python.exe", "-u", "src/server.py"]
    ) is False


def test_looks_like_server_process_matches_venv_python():
    """Venv python paths should still match."""
    assert server_module._looks_like_server_process(
        [".venv/Scripts/python.exe", "-u", "src/server.py"]
    ) is True
    assert server_module._looks_like_server_process(
        [".venv\\Scripts\\python.exe", "-u", "src\\server.py"]
    ) is True
    assert server_module._looks_like_server_process(
        ["C:\\Users\\user\\backend\\.venv\\Scripts\\python.exe", "-u", "src/server.py"]
    ) is True


def test_looks_like_server_process_rejects_inline_scripts():
    """python -c '...src/server.py...' should NOT match."""
    assert server_module._looks_like_server_process(
        ["python", "-c", "import subprocess; subprocess.Popen(['python', '-u', 'src/server.py'])"]
    ) is False
    assert server_module._looks_like_server_process(
        [".venv/Scripts/python.exe", "-c", "from server import main"]
    ) is False


def test_find_conflicting_server_processes_ignores_current_pid(monkeypatch):
    fake_procs = [
        _FakeProc(100, ["python", "-m", "src.server"]),
        _FakeProc(200, ["python", "-m", "src.server"]),
        _FakeProc(300, ["python", "worker.py"]),
    ]

    monkeypatch.setattr(server_module, "psutil", type("P", (), {"process_iter": staticmethod(lambda *_args, **_kwargs: fake_procs)})())

    conflicts = server_module.find_conflicting_server_processes(current_pid=100)

    assert conflicts == [{"pid": 200, "name": "python.exe", "cmdline": ["python", "-m", "src.server"]}]


def test_enforce_single_server_instance_raises_when_conflict_exists(monkeypatch):
    monkeypatch.setattr(
        server_module,
        "find_conflicting_server_processes",
        lambda current_pid=None: [{"pid": 222, "name": "python.exe", "cmdline": ["python", "-m", "src.server"]}],
    )

    with pytest.raises(RuntimeError, match="already running"):
        server_module.enforce_single_server_instance(host="127.0.0.1", port=8000, current_pid=111)


def test_server_instance_lock_rejects_second_holder(tmp_path: Path):
    lock_path = tmp_path / "server.instance.lock"
    first = server_module.ServerInstanceLock(lock_path)
    second = server_module.ServerInstanceLock(lock_path)

    first.acquire(metadata={"host": "127.0.0.1", "port": 8000})
    try:
        with pytest.raises(OSError):
            second.acquire(metadata={"host": "127.0.0.1", "port": 8001})
    finally:
        first.release()
