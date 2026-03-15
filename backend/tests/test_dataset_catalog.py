from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from src.spread.dataset_catalog import catalog_snapshot_manifest, dataset_manifest_path, register_sqlite_dataset


def _write_minimal_snapshot(path: Path) -> Path:
    conn = sqlite3.connect(path)
    try:
        conn.executescript(
            """
            CREATE TABLE tracker_records (
                id INTEGER PRIMARY KEY,
                pair_id INTEGER,
                session_id INTEGER,
                block_id INTEGER,
                ts REAL
            );
            CREATE TABLE tracker_pairs (
                id INTEGER PRIMARY KEY,
                pair TEXT
            );
            CREATE TABLE tracker_pair_blocks (
                id INTEGER PRIMARY KEY
            );
            CREATE TABLE tracker_pair_episodes (
                id INTEGER PRIMARY KEY,
                is_closed INTEGER
            );
            CREATE TABLE tracker_capture_sessions (
                id INTEGER PRIMARY KEY
            );
            """
        )
        conn.executemany("INSERT INTO tracker_pairs(id, pair) VALUES (?, ?)", [(1, "btc"), (2, "eth")])
        conn.executemany("INSERT INTO tracker_pair_blocks(id) VALUES (?)", [(1,), (2,)])
        conn.executemany("INSERT INTO tracker_capture_sessions(id) VALUES (?)", [(10,), (11,)])
        conn.executemany(
            "INSERT INTO tracker_records(pair_id, session_id, block_id, ts) VALUES (?, ?, ?, ?)",
            [
                (1, 10, 1, 1000.0),
                (1, 10, 1, 1010.0),
                (2, 11, 2, 1020.0),
            ],
        )
        conn.executemany(
            "INSERT INTO tracker_pair_episodes(id, is_closed) VALUES (?, ?)",
            [(1, 1), (2, 1), (3, 0)],
        )
        conn.commit()
    finally:
        conn.close()
    return path


def test_register_sqlite_dataset_creates_catalog_and_blessed_copy(tmp_path: Path):
    out_root = tmp_path / "out"
    snapshot_path = _write_minimal_snapshot(tmp_path / "snapshot_test.sqlite")
    cert_path = tmp_path / "snapshot_test.sqlite.cert.json"
    cert_path.write_text(json.dumps({"verdict": "CERTIFIED_WITH_WARNINGS"}), encoding="utf-8")

    entry = register_sqlite_dataset(
        snapshot_path,
        base_path=out_root,
        certification={"verdict": "CERTIFIED_WITH_WARNINGS", "certification_id": "cert-1"},
        label="snapshot_test",
        role="test_snapshot",
        tags=["test", "snapshot"],
        bless=True,
        related_files=[cert_path],
        extra_metadata={"run_id": "run-test"},
    )

    manifest = json.loads(dataset_manifest_path(out_root).read_text(encoding="utf-8"))
    assert manifest["datasets"][0]["dataset_id"] == entry["dataset_id"]
    assert manifest["blessed_dataset_ids"] == [entry["dataset_id"]]
    assert Path(entry["catalog_path"]).is_file()
    assert Path(entry["dataset_manifest_path"]).is_file()
    assert Path(entry["blessed_path"]).is_file()
    assert Path(entry["source_sidecar_path"]).is_file()
    assert entry["record_count"] == 3
    assert entry["pair_count"] == 2
    assert entry["block_count"] == 2
    assert entry["episode_count"] == 3
    assert entry["certification_verdict"] == "WARN"


def test_catalog_snapshot_manifest_imports_existing_snapshots(tmp_path: Path):
    out_root = tmp_path / "out"
    snapshots_dir = out_root / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = _write_minimal_snapshot(snapshots_dir / "snapshot_2026-03-15_08h.sqlite")
    manifest_path = snapshots_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "last_updated_utc": "2026-03-15T08:00:00Z",
                "snapshots": [
                    {
                        "filename": snapshot_path.name,
                        "path": str(Path("snapshots") / snapshot_path.name),
                        "certification_verdict": "WARN",
                        "certification_raw_verdict": "CERTIFIED_WITH_WARNINGS",
                        "certification_id": "cert-abc",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    summary = catalog_snapshot_manifest(base_path=out_root)

    manifest = json.loads(dataset_manifest_path(out_root).read_text(encoding="utf-8"))
    assert summary["cataloged_count"] == 1
    assert len(manifest["datasets"]) == 1
    assert manifest["datasets"][0]["label"] == "snapshot_2026-03-15_08h"
    assert manifest["datasets"][0]["blessed"] is True
