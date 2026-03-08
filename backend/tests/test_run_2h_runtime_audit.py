from pathlib import Path

from tests import run_2h_runtime_audit


def test_runtime_audit_runner_resolves_paths_absolutely():
    output_dir, db_path = run_2h_runtime_audit.resolve_runtime_paths(
        "out/runtime_audit/demo",
        "out/config/demo.sqlite",
    )

    assert output_dir.is_absolute()
    assert db_path.is_absolute()
    assert str(output_dir).endswith(str(Path("backend") / "out" / "runtime_audit" / "demo"))
    assert str(db_path).endswith(str(Path("backend") / "out" / "config" / "demo.sqlite"))


def test_runtime_audit_runner_finalizes_with_operator_cancelled_on_keyboard_interrupt(monkeypatch, tmp_path: Path):
    finalized: dict[str, object] = {}

    monkeypatch.setattr(
        run_2h_runtime_audit,
        "resolve_runtime_paths",
        lambda output_dir_arg, db_path_arg: (tmp_path / "audit", tmp_path / "tracker.sqlite"),
    )
    monkeypatch.setattr(run_2h_runtime_audit.time, "sleep", lambda _: None)

    state = {"calls": 0}

    def _interrupting_probe(base_url: str, output_path: Path) -> None:
        state["calls"] += 1
        if state["calls"] >= 1:
            raise KeyboardInterrupt()

    monkeypatch.setattr(run_2h_runtime_audit, "_probe_dashboard", _interrupting_probe)

    def _fake_finalize_runtime_audit_package(**kwargs):
        finalized.update(kwargs)
        return {"summary_path": str(tmp_path / "audit" / "summary.json")}

    monkeypatch.setattr(run_2h_runtime_audit, "finalize_runtime_audit_package", _fake_finalize_runtime_audit_package)
    monkeypatch.setattr(
        run_2h_runtime_audit.sys,
        "argv",
        ["run_2h_runtime_audit.py", "--duration-sec", "5"],
    )

    run_2h_runtime_audit.main()

    assert finalized["run_status"] == "operator_cancelled"
