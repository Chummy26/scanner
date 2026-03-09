import sqlite3
from pathlib import Path

from tests import run_server_data_hygiene_certification


def test_certification_runner_resolves_paths_absolutely(tmp_path: Path):
    output_dir, canonical_db = run_server_data_hygiene_certification.resolve_certification_paths(
        "out/certification",
        "out/config/tracker.sqlite",
    )

    assert output_dir.is_absolute()
    assert canonical_db.is_absolute()
    assert str(output_dir).endswith(str(Path("out") / "certification"))
    assert str(canonical_db).endswith(str(Path("backend") / "out" / "config" / "tracker.sqlite"))


def test_clone_db_family_copies_sqlite_sidecars(tmp_path: Path):
    source_db = tmp_path / "source.sqlite"
    sqlite3.connect(source_db).close()
    for suffix in ("-wal", "-shm"):
        Path(f"{source_db}{suffix}").write_text("x", encoding="utf-8")
    target_db = tmp_path / "clone" / "tracker.sqlite"

    copied = run_server_data_hygiene_certification.clone_db_family(source_db, target_db)

    assert Path(copied[0]).is_file()
    assert Path(f"{target_db}-wal").is_file()
    assert Path(f"{target_db}-shm").is_file()


def test_collect_sqlite_integrity_handles_missing_db(tmp_path: Path):
    payload = run_server_data_hygiene_certification.collect_sqlite_integrity(tmp_path / "missing.sqlite")

    assert payload["db_exists"] is False
    assert payload["anomalies"]["zero_record_blocks"] == 0


def test_certification_report_mentions_both_tracks():
    report = run_server_data_hygiene_certification.build_report(
        {
            "baseline": {"ok": True, "command": "python -m pytest tests -q"},
            "tracks": {
                "current_clone": {
                    "pre_integrity": {"anomalies": {"zero_record_blocks": 1}},
                    "post_integrity": {"anomalies": {"zero_record_blocks": 1}},
                    "runtime": {"verify_runtime": {"quality_summary": {"critical_sessions": 1, "training_ready_sessions": 0}}},
                    "dataset_matrix": [{}, {}],
                },
                "clean_cycle": {
                    "post_integrity": {"anomalies": {"zero_record_blocks": 0}},
                    "runtime": {"verify_runtime": {"quality_summary": {"critical_sessions": 0, "training_ready_sessions": 1}}, "checkpoints": [{}, {}]},
                    "dataset_matrix": [{}, {}, {}],
                },
            },
            "reset_smoke": {"archive_dir": "x"},
        }
    )

    assert "# Server Data Hygiene Certification" in report
    assert "## Current Clone" in report
    assert "## Clean Cycle" in report
    assert "## Reset Smoke" in report


def test_short_current_clone_skips_heavy_postprocessing(monkeypatch, tmp_path: Path):
    output_dir = tmp_path / "certification_short"
    canonical_db = tmp_path / "canonical.sqlite"
    sqlite3.connect(canonical_db).close()
    dataset_calls: list[str] = []
    preflight_calls: list[str] = []

    monkeypatch.setattr(
        run_server_data_hygiene_certification,
        "resolve_certification_paths",
        lambda *_args, **_kwargs: (output_dir, canonical_db),
    )
    monkeypatch.setattr(
        run_server_data_hygiene_certification,
        "clone_db_family",
        lambda source_db, target_db: [str(target_db)],
    )
    monkeypatch.setattr(
        run_server_data_hygiene_certification,
        "collect_sqlite_integrity",
        lambda _db_path: {"anomalies": {"zero_record_blocks": 0}},
    )
    monkeypatch.setattr(
        run_server_data_hygiene_certification,
        "run_server_track",
        lambda **kwargs: {"verify_runtime": {"quality_summary": {}}, "checkpoints": []},
    )
    monkeypatch.setattr(
        run_server_data_hygiene_certification,
        "collect_dataset_matrix",
        lambda db_path, configs=None: dataset_calls.append(str(Path(db_path).parent.name)) or [{"ok": True}],
    )
    monkeypatch.setattr(
        run_server_data_hygiene_certification,
        "collect_threshold_preflights",
        lambda db_path, configs=None: preflight_calls.append(str(Path(db_path).parent.name)) or [{"ok": True}],
    )
    monkeypatch.setattr(
        run_server_data_hygiene_certification,
        "_run_baseline_tests",
        lambda: {"ok": True, "command": "skipped", "stdout": "", "stderr": "", "returncode": 0},
    )
    monkeypatch.setattr(
        run_server_data_hygiene_certification.sys,
        "argv",
        [
            "run_server_data_hygiene_certification.py",
            "--output-dir",
            str(output_dir),
            "--canonical-db",
            str(canonical_db),
            "--current-duration-sec",
            "60",
            "--clean-duration-sec",
            "300",
            "--skip-baseline-tests",
            "--skip-reset-smoke",
        ],
    )

    run_server_data_hygiene_certification.main()

    assert dataset_calls == ["clean_cycle"]
    assert preflight_calls == ["clean_cycle"]
