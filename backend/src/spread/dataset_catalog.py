from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def backend_out_root(base_path: Path | None = None) -> Path:
    return Path(base_path) if base_path is not None else Path(__file__).resolve().parent.parent.parent / "out"


def datasets_dir(base_path: Path | None = None) -> Path:
    return backend_out_root(base_path) / "datasets"


def dataset_registry_dir(base_path: Path | None = None) -> Path:
    return datasets_dir(base_path) / "registry"


def dataset_blessed_dir(base_path: Path | None = None) -> Path:
    return datasets_dir(base_path) / "blessed"


def dataset_manifest_path(base_path: Path | None = None) -> Path:
    return datasets_dir(base_path) / "manifest.json"


def snapshot_sidecar_path(snapshot_path: Path) -> Path:
    return Path(f"{Path(snapshot_path)}.dataset_manifest.json")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def read_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _slugify(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(value).strip()).strip("-_.").lower()
    return normalized or "dataset"


def normalize_certification_verdict(raw_value: Any) -> str:
    normalized = str(raw_value or "").strip().upper()
    if normalized in {"CERTIFIED", "PASS"}:
        return "PASS"
    if normalized in {"CERTIFIED_WITH_WARNINGS", "WARN", "WARNING"}:
        return "WARN"
    if normalized in {"FAILED", "FAIL"}:
        return "FAIL"
    return normalized or "UNKNOWN"


def _sqlite_table_counts(sqlite_path: Path) -> dict[str, int]:
    with sqlite3.connect(sqlite_path, timeout=30.0) as conn:
        return {
            "tracker_records": int(conn.execute("SELECT COUNT(*) FROM tracker_records").fetchone()[0]),
            "tracker_pairs": int(conn.execute("SELECT COUNT(*) FROM tracker_pairs").fetchone()[0]),
            "tracker_pair_blocks": int(conn.execute("SELECT COUNT(*) FROM tracker_pair_blocks").fetchone()[0]),
            "tracker_pair_episodes": int(conn.execute("SELECT COUNT(*) FROM tracker_pair_episodes").fetchone()[0]),
        }


def _sqlite_time_range(sqlite_path: Path) -> tuple[float, float]:
    with sqlite3.connect(sqlite_path, timeout=30.0) as conn:
        row = conn.execute("SELECT MIN(ts), MAX(ts) FROM tracker_records").fetchone()
    return float(row[0] or 0.0), float(row[1] or 0.0)


def _sqlite_session_ids(sqlite_path: Path) -> list[int]:
    with sqlite3.connect(sqlite_path, timeout=30.0) as conn:
        rows = conn.execute("SELECT DISTINCT id FROM tracker_capture_sessions ORDER BY id ASC").fetchall()
    return [int(row[0]) for row in rows if int(row[0] or 0) > 0]


def describe_sqlite_dataset(
    sqlite_path: Path,
    *,
    certification: dict[str, Any] | None = None,
    label: str | None = None,
    role: str = "snapshot",
    tags: list[str] | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sqlite_path = Path(sqlite_path)
    table_counts = _sqlite_table_counts(sqlite_path)
    time_range = _sqlite_time_range(sqlite_path)
    sha256 = sha256_file(sqlite_path)
    dataset_id = f"{_slugify(label or sqlite_path.stem)}-{sha256[:12]}"
    cert_payload = dict(certification or {})
    return {
        "dataset_id": dataset_id,
        "label": str(label or sqlite_path.stem),
        "role": str(role),
        "kind": "sqlite_snapshot",
        "filename": sqlite_path.name,
        "source_path": str(sqlite_path.resolve()),
        "size_bytes": int(sqlite_path.stat().st_size if sqlite_path.exists() else 0),
        "sha256": sha256,
        "record_time_range": [float(time_range[0]), float(time_range[1])],
        "record_count": int(table_counts.get("tracker_records", 0)),
        "pair_count": int(table_counts.get("tracker_pairs", 0)),
        "block_count": int(table_counts.get("tracker_pair_blocks", 0)),
        "episode_count": int(table_counts.get("tracker_pair_episodes", 0)),
        "session_ids": _sqlite_session_ids(sqlite_path),
        "certification_verdict": normalize_certification_verdict(
            cert_payload.get("certification_verdict") or cert_payload.get("verdict")
        ),
        "certification_raw_verdict": str(
            cert_payload.get("certification_raw_verdict") or cert_payload.get("verdict") or ""
        ),
        "certification_id": str(cert_payload.get("certification_id") or ""),
        "certification_failure_reasons": [str(value) for value in (cert_payload.get("failure_reasons") or [])],
        "certification_warnings": [str(value) for value in (cert_payload.get("warnings") or [])],
        "tags": sorted({str(tag) for tag in (tags or []) if str(tag).strip()}),
        "extra_metadata": dict(extra_metadata or {}),
    }


def describe_state_file(state_file: Path) -> dict[str, Any]:
    state_path = Path(state_file)
    payload = {
        "state_file": str(state_path.resolve()),
        "state_file_name": state_path.name,
        "state_file_exists": bool(state_path.exists()),
        "state_file_size_bytes": int(state_path.stat().st_size if state_path.exists() else 0),
        "state_file_kind": "sqlite" if state_path.suffix.lower() == ".sqlite" else "file",
        "state_file_sha256": "",
        "dataset_manifest_path": "",
    }
    if not state_path.exists():
        return payload
    payload["state_file_sha256"] = sha256_file(state_path)
    sidecar_path = snapshot_sidecar_path(state_path)
    if sidecar_path.is_file():
        payload["dataset_manifest_path"] = str(sidecar_path.resolve())
    if state_path.suffix.lower() == ".sqlite":
        payload.update(
            {
                "record_time_range": list(_sqlite_time_range(state_path)),
                "session_ids": _sqlite_session_ids(state_path),
                "table_counts": _sqlite_table_counts(state_path),
            }
        )
    return payload


def _materialize_file(source_path: Path, target_path: Path) -> str:
    source = Path(source_path)
    target = Path(target_path)
    if source.resolve() == target.resolve():
        return "in_place"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    try:
        os.link(str(source), str(target))
        return "hardlink"
    except Exception:
        shutil.copy2(source, target)
        return "copy"


def _upsert_dataset_manifest(entry: dict[str, Any], *, base_path: Path | None = None) -> dict[str, Any]:
    manifest = read_json(dataset_manifest_path(base_path), {"datasets": [], "last_updated_utc": "", "blessed_dataset_ids": []})
    datasets = [item for item in list(manifest.get("datasets") or []) if str(item.get("dataset_id") or "") != str(entry.get("dataset_id") or "")]
    datasets.append(dict(entry))
    datasets.sort(key=lambda item: (str(item.get("label") or ""), str(item.get("dataset_id") or "")))
    manifest["datasets"] = datasets
    blessed_ids = {
        str(item.get("dataset_id") or "")
        for item in datasets
        if bool(item.get("blessed"))
    }
    manifest["blessed_dataset_ids"] = sorted(value for value in blessed_ids if value)
    manifest["last_updated_utc"] = _utc_now_iso()
    write_json(dataset_manifest_path(base_path), manifest)
    return manifest


def register_sqlite_dataset(
    sqlite_path: Path,
    *,
    base_path: Path | None = None,
    certification: dict[str, Any] | None = None,
    label: str | None = None,
    role: str = "snapshot",
    tags: list[str] | None = None,
    bless: bool = False,
    related_files: list[Path] | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source_path = Path(sqlite_path).resolve()
    descriptor = describe_sqlite_dataset(
        source_path,
        certification=certification,
        label=label,
        role=role,
        tags=tags,
        extra_metadata=extra_metadata,
    )
    dataset_id = str(descriptor["dataset_id"])
    registry_root = dataset_registry_dir(base_path)
    registry_dir = registry_root / dataset_id
    registry_file = registry_dir / source_path.name
    materialization_method = _materialize_file(source_path, registry_file)
    copied_related: list[dict[str, Any]] = []
    for related in list(related_files or []):
        related_path = Path(related)
        if not related_path.is_file():
            continue
        target_related = registry_dir / related_path.name
        copied_related.append(
            {
                "source_path": str(related_path.resolve()),
                "path": str(target_related.resolve()),
                "materialization": _materialize_file(related_path, target_related),
            }
        )

    entry = {
        **descriptor,
        "catalog_dir": str(registry_dir.resolve()),
        "catalog_path": str(registry_file.resolve()),
        "dataset_manifest_path": str((registry_dir / "dataset_manifest.json").resolve()),
        "source_sidecar_path": str(snapshot_sidecar_path(source_path).resolve()),
        "blessed": bool(bless),
        "materialization": {"catalog": materialization_method},
        "related_files": copied_related,
        "updated_at_utc": _utc_now_iso(),
    }

    if bless:
        blessed_root = dataset_blessed_dir(base_path) / dataset_id
        blessed_file = blessed_root / source_path.name
        entry["blessed_dir"] = str(blessed_root.resolve())
        entry["blessed_path"] = str(blessed_file.resolve())
        entry["materialization"]["blessed"] = _materialize_file(source_path, blessed_file)
        if copied_related:
            for item in copied_related:
                related_source = Path(str(item["source_path"]))
                target_related = blessed_root / related_source.name
                _materialize_file(related_source, target_related)

    write_json(registry_dir / "dataset_manifest.json", entry)
    write_json(snapshot_sidecar_path(source_path), entry)
    _upsert_dataset_manifest(entry, base_path=base_path)
    return entry


def catalog_snapshot_manifest(*, base_path: Path | None = None) -> dict[str, Any]:
    root = backend_out_root(base_path)
    snapshot_manifest = read_json(root / "snapshots" / "manifest.json", {"snapshots": []})
    results: list[dict[str, Any]] = []
    for item in list(snapshot_manifest.get("snapshots") or []):
        snapshot_path = Path(str(item.get("path") or ""))
        if not snapshot_path.is_absolute():
            candidates = [
                (root / snapshot_path).resolve(),
                (root.parent / snapshot_path).resolve(),
                (root.parent.parent / snapshot_path).resolve(),
                (Path.cwd() / snapshot_path).resolve(),
            ]
            snapshot_path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
        if not snapshot_path.is_file() or snapshot_path.suffix.lower() != ".sqlite":
            continue
        certification_payload = {
            "verdict": item.get("certification_raw_verdict") or item.get("certification_verdict"),
            "certification_id": item.get("certification_id"),
        }
        cert_sidecar = Path(str(snapshot_path) + ".cert.json")
        related_files = [cert_sidecar] if cert_sidecar.is_file() else []
        entry = register_sqlite_dataset(
            snapshot_path,
            base_path=root,
            certification=certification_payload,
            label=str(item.get("filename") or snapshot_path.stem).replace(".sqlite", ""),
            role="scheduled_snapshot",
            tags=["snapshot", "cataloged"],
            bless=normalize_certification_verdict(item.get("certification_verdict")) in {"PASS", "WARN"},
            related_files=related_files,
            extra_metadata={
                "created_at_utc": item.get("created_at_utc"),
                "source_snapshot_manifest_path": str((root / "snapshots" / "manifest.json").resolve()),
            },
        )
        results.append(entry)
    summary = {
        "cataloged_count": len(results),
        "dataset_manifest_path": str(dataset_manifest_path(root).resolve()),
        "cataloged_dataset_ids": [str(item.get("dataset_id") or "") for item in results],
    }
    write_json(datasets_dir(root) / "catalog_sync_summary.json", summary)
    return summary
