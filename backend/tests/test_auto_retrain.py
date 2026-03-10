from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from src.spread.auto_retrain import (
    current_snapshot_slot,
    next_snapshot_slot,
    should_retrain,
    update_snapshot_manifest,
    write_json,
)


def _seed_snapshot_sqlite(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE tracker_records(pair_id INTEGER PRIMARY KEY, ts REAL NOT NULL, entry_spread_pct REAL NOT NULL, exit_spread_pct REAL NOT NULL);
            CREATE TABLE tracker_pairs(id INTEGER PRIMARY KEY, symbol TEXT NOT NULL);
            CREATE TABLE tracker_pair_blocks(id INTEGER PRIMARY KEY, pair_id INTEGER NOT NULL, session_id INTEGER NOT NULL);
            CREATE TABLE tracker_pair_episodes(id INTEGER PRIMARY KEY, pair_id INTEGER NOT NULL, session_id INTEGER NOT NULL, block_id INTEGER NOT NULL, start_ts REAL NOT NULL, peak_ts REAL NOT NULL, end_ts REAL NOT NULL, duration_sec REAL NOT NULL, peak_entry_spread REAL NOT NULL, exit_spread_at_close REAL NOT NULL, baseline_median REAL NOT NULL, baseline_mad REAL NOT NULL, activation_threshold REAL NOT NULL, release_threshold REAL NOT NULL, source_version TEXT NOT NULL, is_closed INTEGER NOT NULL);
            CREATE TABLE tracker_capture_sessions(id INTEGER PRIMARY KEY, started_at REAL NOT NULL);
            """
        )
        conn.execute("INSERT INTO tracker_pairs(id, symbol) VALUES(1, 'BTC')")
        conn.execute("INSERT INTO tracker_capture_sessions(id, started_at) VALUES(1, 100.0)")
        conn.execute("INSERT INTO tracker_pair_blocks(id, pair_id, session_id) VALUES(1, 1, 1)")
        conn.execute("INSERT INTO tracker_records(pair_id, ts, entry_spread_pct, exit_spread_pct) VALUES(1, 100.0, 0.5, -0.1)")
        conn.execute(
            """
            INSERT INTO tracker_pair_episodes(
                id, pair_id, session_id, block_id, start_ts, peak_ts, end_ts, duration_sec,
                peak_entry_spread, exit_spread_at_close, baseline_median, baseline_mad,
                activation_threshold, release_threshold, source_version, is_closed
            )
            VALUES(1, 1, 1, 1, 100.0, 110.0, 120.0, 20.0, 1.2, 0.2, 0.5, 0.1, 0.7, 0.4, 'recurring_v1', 1)
            """
        )


def test_snapshot_slot_helpers_use_fixed_utc_boundaries():
    morning = datetime(2026, 3, 10, 7, 50, tzinfo=timezone.utc)
    assert current_snapshot_slot(morning) == datetime(2026, 3, 10, 0, 0, tzinfo=timezone.utc)
    assert next_snapshot_slot(morning) == datetime(2026, 3, 10, 8, 0, tzinfo=timezone.utc)

    after_close = datetime(2026, 3, 10, 16, 1, tzinfo=timezone.utc)
    assert current_snapshot_slot(after_close) == datetime(2026, 3, 10, 16, 0, tzinfo=timezone.utc)
    assert next_snapshot_slot(after_close) == datetime(2026, 3, 11, 0, 0, tzinfo=timezone.utc)


def test_update_snapshot_manifest_writes_integrity_metadata(tmp_path: Path):
    base_path = tmp_path / "out"
    snapshot_path = base_path / "snapshots" / "snapshot_2026-03-10_08h.sqlite"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    _seed_snapshot_sqlite(snapshot_path)

    manifest = update_snapshot_manifest(
        snapshot_path,
        {"verdict": "CERTIFIED", "certification_id": "sha1_test"},
        base_path=base_path,
    )

    assert manifest["snapshots"][0]["filename"] == snapshot_path.name
    assert manifest["snapshots"][0]["certification_verdict"] == "PASS"
    assert manifest["snapshots"][0]["record_count"] == 1
    assert manifest["snapshots"][0]["pair_count"] == 1
    assert manifest["snapshots"][0]["episode_count"] == 1
    assert manifest["snapshots"][0]["sha256"]


def test_should_retrain_uses_snapshot_and_training_manifests(tmp_path: Path):
    base_path = tmp_path / "out"
    base_ts = 10_000.0
    snapshots = [
        {"filename": f"snapshot_{index}.sqlite", "created_at_utc_ts": base_ts + index, "certification_verdict": "PASS"}
        for index in range(4)
    ]
    write_json(base_path / "snapshots" / "manifest.json", {"snapshots": snapshots})

    with patch("src.spread.auto_retrain.time.time", return_value=base_ts + 10.0):
        assert should_retrain(base_path=base_path, model_dir=base_path / "config") == "first_training"

        write_json(
            base_path / "training" / "manifest.json",
            {"runs": [{"run_id": "run_1", "created_at_ts": base_ts, "finished_at_ts": base_ts}]},
        )
        assert should_retrain(base_path=base_path, model_dir=base_path / "config") == "new_data_available"
