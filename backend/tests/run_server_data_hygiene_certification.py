import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.spread.ml_dataset import build_dataset_bundle, build_group_splits
from src.spread.training_certification import collect_sqlite_integrity as collect_sqlite_integrity_lib
from src.spread.runtime_audit import _write_json, build_runtime_summary
from src.spread.train_model import run_threshold_preflight
from tests import run_10m_diagnostic, run_2h_runtime_audit, verify_training_blocks_runtime

DEFAULT_DATASET_CONFIGS = [
    {"sequence_length": 4, "prediction_horizon_sec": 240},
    {"sequence_length": 4, "prediction_horizon_sec": 600},
    {"sequence_length": 8, "prediction_horizon_sec": 240},
    {"sequence_length": 8, "prediction_horizon_sec": 600},
    {"sequence_length": 15, "prediction_horizon_sec": 14_400},
]
DEFAULT_PREFLIGHT_CONFIGS = [
    {"sequence_length": 4, "prediction_horizon_sec": 240, "thresholds": [0.0, 0.3, 0.5, 0.8, 1.0]},
    {"sequence_length": 8, "prediction_horizon_sec": 600, "thresholds": [0.0, 0.3, 0.5, 0.8, 1.0]},
    {"sequence_length": 15, "prediction_horizon_sec": 14_400, "thresholds": [0.0, 0.3, 0.5, 0.8, 1.0]},
]


def resolve_certification_paths(output_dir_arg: str, canonical_db_arg: str) -> tuple[Path, Path]:
    output_dir = Path(output_dir_arg) if output_dir_arg else (ROOT_DIR / "out" / "server_data_hygiene")
    if not output_dir.is_absolute():
        output_dir = (Path.cwd() / output_dir).resolve()
    canonical_db = Path(canonical_db_arg)
    if not canonical_db.is_absolute():
        canonical_db = (ROOT_DIR / canonical_db).resolve()
    return output_dir, canonical_db


def clone_db_family(source_db: Path, target_db: Path) -> list[str]:
    target_db.parent.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for suffix in ("", "-wal", "-shm"):
        source_path = Path(f"{source_db}{suffix}")
        if not source_path.exists():
            continue
        target_path = Path(f"{target_db}{suffix}")
        shutil.copy2(source_path, target_path)
        copied.append(str(target_path))
    return copied


def _table_count(conn: sqlite3.Connection, table_name: str) -> int:
    row = conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    if row is None or int(row[0]) <= 0:
        return 0
    return int(conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0])


def collect_sqlite_integrity(db_path: Path) -> dict[str, object]:
    return collect_sqlite_integrity_lib(db_path)


def collect_dataset_matrix(db_path: Path, configs: list[dict[str, int]] | None = None) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for config in configs or DEFAULT_DATASET_CONFIGS:
        sequence_length = int(config["sequence_length"])
        prediction_horizon_sec = int(config["prediction_horizon_sec"])
        try:
            bundle = build_dataset_bundle(
                db_path,
                sequence_length=sequence_length,
                prediction_horizon_sec=prediction_horizon_sec,
            )
            split_summary = {}
            if int(bundle.summary.get("num_samples", 0)) > 0:
                split_summary = build_group_splits(
                    bundle,
                    prediction_horizon_sec=prediction_horizon_sec,
                )["train"].summary.get("split_summary", {})
            results.append(
                {
                    "sequence_length": sequence_length,
                    "prediction_horizon_sec": prediction_horizon_sec,
                    "num_samples": int(bundle.summary.get("num_samples", 0)),
                    "num_positive_samples": int(bundle.summary.get("num_positive_samples", 0)),
                    "right_censored_windows": int(bundle.summary.get("skipped_windows_right_censored", 0)),
                    "eligible_blocks": int(
                        bundle.summary.get("block_diagnostics", {})
                        .get("feature_window_feasibility", {})
                        .get("eligible_blocks_for_sequence_length", 0)
                    ),
                    "eligible_sessions": int(
                        bundle.summary.get("block_diagnostics", {})
                        .get("feature_window_feasibility", {})
                        .get("eligible_sessions_with_any_eligible_block", 0)
                    ),
                    "purged_temporal_separation_ok": bool(split_summary.get("purged_temporal_separation_ok", False)),
                    "split_mode": str(split_summary.get("split_mode", "")),
                }
            )
        except Exception as exc:
            results.append(
                {
                    "sequence_length": sequence_length,
                    "prediction_horizon_sec": prediction_horizon_sec,
                    "error": str(exc),
                }
            )
    return results


def collect_threshold_preflights(db_path: Path, configs: list[dict[str, object]] | None = None) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for config in configs or DEFAULT_PREFLIGHT_CONFIGS:
        merge_off = run_threshold_preflight(
            state_file=db_path,
            sequence_length=int(config["sequence_length"]),
            prediction_horizon_sec=int(config["prediction_horizon_sec"]),
            thresholds=[float(value) for value in config["thresholds"]],
            allow_cross_session_merge=False,
        )
        merge_on = run_threshold_preflight(
            state_file=db_path,
            sequence_length=int(config["sequence_length"]),
            prediction_horizon_sec=int(config["prediction_horizon_sec"]),
            thresholds=[float(value) for value in config["thresholds"]],
            allow_cross_session_merge=True,
        )
        results.append(
            {
                "sequence_length": int(config["sequence_length"]),
                "prediction_horizon_sec": int(config["prediction_horizon_sec"]),
                "merge_off": merge_off,
                "merge_on": merge_on,
            }
        )
    return results


def _fetch_json(base_url: str, path: str) -> dict:
    with urllib.request.urlopen(f"{base_url.rstrip('/')}{path}", timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _post_json(base_url: str, path: str, payload: dict | None = None) -> dict:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=json.dumps(payload or {}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _wait_for_api_down(base_url: str, timeout_sec: int = 30) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"{base_url.rstrip('/')}/api/v1/ml/dashboard", timeout=3)
        except Exception:
            return True
        time.sleep(1)
    return False


def _terminate_process_tree(process: subprocess.Popen | None, *, base_url: str = "") -> None:
    if process is None:
        return
    try:
        if process.poll() is not None:
            if base_url:
                _wait_for_api_down(base_url, timeout_sec=5)
            return
    except Exception:
        pass
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        try:
            process.wait(timeout=10)
        except Exception:
            pass
    else:
        process.terminate()
        try:
            process.wait(timeout=30)
        except Exception:
            process.kill()
    if base_url:
        _wait_for_api_down(base_url, timeout_sec=30)


def _build_lightweight_runtime_package(
    *,
    output_dir: Path,
    duration_sec: int,
    run_status: str,
) -> dict[str, object]:
    runtime_package: dict[str, object] = {}
    try:
        audit_dir = output_dir / "runtime_audit"
        if (audit_dir / "events.ndjson").is_file():
            ws_latency_path = audit_dir / "ws_latency_summary.json"
            ws_latency_summary = None
            if ws_latency_path.is_file():
                try:
                    ws_latency_summary = json.loads(ws_latency_path.read_text(encoding="utf-8"))
                except Exception:
                    ws_latency_summary = None
            summary = build_runtime_summary(audit_dir, ws_latency_summary=ws_latency_summary)
            summary["finished_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            summary["duration_sec"] = int(duration_sec)
            summary["run_status"] = str(run_status)
            _write_json(audit_dir / "summary.json", summary)
            runtime_package = {"path": str(audit_dir), "summary": summary}
        else:
            runtime_package = {"path": str(audit_dir), "summary": {}}
    except Exception as exc:
        runtime_package = {"path": str(output_dir / "runtime_audit"), "error": str(exc)}
    return runtime_package


def collect_runtime_checkpoint(base_url: str, *, sequence_horizon_pairs: list[tuple[int, int]]) -> dict[str, object]:
    checkpoint = {
        "dashboard_summary": _fetch_json(base_url, "/api/v1/ml/dashboard").get("summary", {}),
        "sessions_summary": _fetch_json(base_url, "/api/v1/ml/training/sessions?include_open=1&summary_only=1").get("summary", {}),
        "blocks_summary": _fetch_json(base_url, "/api/v1/ml/training/blocks?summary_only=1").get("summary", {}),
        "cohort_preview_summary": _post_json(base_url, "/api/v1/ml/training/cohorts/preview", {"summary_only": True}).get("summary", {}),
        "quality_reports": [],
    }
    for sequence_length, prediction_horizon_sec in sequence_horizon_pairs:
        quality_payload = _fetch_json(
            base_url,
            f"/api/v1/ml/training/quality-report?sequence_length={int(sequence_length)}&prediction_horizon_sec={int(prediction_horizon_sec)}&summary_only=1",
        )
        checkpoint["quality_reports"].append(
            {
                "sequence_length": int(sequence_length),
                "prediction_horizon_sec": int(prediction_horizon_sec),
                "summary": quality_payload.get("summary", {}),
            }
        )
    return checkpoint


def _run_baseline_tests() -> dict[str, object]:
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "-q"],
        cwd=str(ROOT_DIR),
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "command": "python -m pytest tests -q",
        "returncode": int(completed.returncode),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "ok": completed.returncode == 0,
    }


def run_server_track(
    *,
    track_name: str,
    db_path: Path,
    output_dir: Path,
    duration_sec: int,
    checkpoint_seconds: list[int],
    base_url: str,
    probe_interval_sec: int,
    max_symbols: int,
    symbol_discovery_enabled: int,
    diagnostic_duration_sec: int = 0,
    run_reset_smoke: bool = False,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["TEAM_OP_RUNTIME_AUDIT_DIR"] = str(output_dir / "runtime_audit")
    env["TEAM_OP_RUNTIME_AUDIT_DURATION_SEC"] = str(int(duration_sec))
    env["TEAM_OP_TRACKER_DB_PATH"] = str(db_path)
    env["TEAM_OP_TRACKER_CAPTURE_MODE"] = "continuous_all_pairs"
    env["TEAM_OP_MAX_SYMBOLS"] = str(int(max_symbols))
    env["TEAM_OP_SYMBOL_DISCOVERY_ENABLED"] = str(int(symbol_discovery_enabled))
    server_process = subprocess.Popen(
        [sys.executable, "src/server.py"],
        cwd=str(ROOT_DIR),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    checkpoints: list[dict[str, object]] = []
    api_probe_path = output_dir / "api_probe.ndjson"
    run_status = "completed"
    diagnostic_report_path = output_dir / "runtime_diagnostic.md"
    try:
        if not run_2h_runtime_audit._wait_for_api(base_url, timeout_sec=45):
            raise RuntimeError(f"{track_name}: server did not become ready in time")
        started = time.time()
        pending_checkpoints = sorted(int(value) for value in checkpoint_seconds if int(value) > 0)
        while (time.time() - started) < float(duration_sec):
            run_2h_runtime_audit._probe_dashboard(base_url, api_probe_path)
            elapsed_sec = int(time.time() - started)
            while pending_checkpoints and elapsed_sec >= pending_checkpoints[0]:
                checkpoint_value = pending_checkpoints.pop(0)
                checkpoints.append(
                    {
                        "elapsed_sec": int(checkpoint_value),
                        **collect_runtime_checkpoint(
                            base_url,
                            sequence_horizon_pairs=[(4, 240), (8, 600), (15, 14_400)],
                        ),
                    }
                )
            time.sleep(max(1, int(probe_interval_sec)))
        verify_runtime = verify_training_blocks_runtime.inspect_runtime(db_path, base_url)
        verify_report = verify_training_blocks_runtime.build_report(verify_runtime)
        (output_dir / "training_runtime_report.md").write_text(verify_report, encoding="utf-8")
        diagnostic_metrics = {}
        if int(diagnostic_duration_sec) > 0:
            diagnostic_metrics = run_10m_diagnostic.collect_runtime_metrics(
                api_url=f"{base_url.rstrip('/')}/api/v1/ml/dashboard",
                db_path=db_path,
                duration_sec=int(diagnostic_duration_sec),
                probe_interval_sec=max(1, int(probe_interval_sec)),
            )
            diagnostic_report_path.write_text(run_10m_diagnostic.build_report(diagnostic_metrics), encoding="utf-8")
        reset_smoke = {}
        if run_reset_smoke:
            reset_smoke = _post_json(base_url, "/api/v1/ml/training/reset-cycle", {})
        _terminate_process_tree(server_process, base_url=base_url)
        server_process = None
        runtime_package = _build_lightweight_runtime_package(
            output_dir=output_dir,
            duration_sec=int(duration_sec),
            run_status=run_status,
        )
        return {
            "track_name": track_name,
            "runtime_package": runtime_package,
            "verify_runtime": verify_runtime,
            "checkpoints": checkpoints,
            "diagnostic_metrics": diagnostic_metrics,
            "diagnostic_report_path": str(diagnostic_report_path) if diagnostic_metrics else "",
            "reset_smoke": reset_smoke,
        }
    finally:
        _terminate_process_tree(server_process, base_url=base_url)


def build_report(certification: dict[str, object]) -> str:
    baseline = certification.get("baseline", {})
    current_track = certification.get("tracks", {}).get("current_clone", {})
    clean_track = certification.get("tracks", {}).get("clean_cycle", {})
    return (
        "# Server Data Hygiene Certification\n\n"
        "## Baseline\n"
        f"- Test suite ok: {bool(baseline.get('ok'))}\n"
        f"- Command: {baseline.get('command', '')}\n\n"
        "## Current Clone\n"
        f"- Pre integrity anomalies: {current_track.get('pre_integrity', {}).get('anomalies', {})}\n"
        f"- Post integrity anomalies: {current_track.get('post_integrity', {}).get('anomalies', {})}\n"
        f"- Runtime critical sessions: {current_track.get('runtime', {}).get('verify_runtime', {}).get('quality_summary', {}).get('critical_sessions', 0)}\n"
        f"- Runtime training-ready sessions: {current_track.get('runtime', {}).get('verify_runtime', {}).get('quality_summary', {}).get('training_ready_sessions', 0)}\n"
        f"- Dataset matrix rows: {len(current_track.get('dataset_matrix', []))}\n\n"
        "## Clean Cycle\n"
        f"- Post integrity anomalies: {clean_track.get('post_integrity', {}).get('anomalies', {})}\n"
        f"- Runtime critical sessions: {clean_track.get('runtime', {}).get('verify_runtime', {}).get('quality_summary', {}).get('critical_sessions', 0)}\n"
        f"- Runtime training-ready sessions: {clean_track.get('runtime', {}).get('verify_runtime', {}).get('quality_summary', {}).get('training_ready_sessions', 0)}\n"
        f"- Dataset matrix rows: {len(clean_track.get('dataset_matrix', []))}\n"
        f"- Checkpoints captured: {len(clean_track.get('runtime', {}).get('checkpoints', []))}\n\n"
        "## Reset Smoke\n"
        f"- Executed: {bool(certification.get('reset_smoke'))}\n"
        f"- Result keys: {sorted((certification.get('reset_smoke') or {}).keys())}\n"
    )


def main():
    parser = argparse.ArgumentParser(description="Run two-track server data hygiene certification.")
    parser.add_argument("--output-dir", default="out/server_data_hygiene")
    parser.add_argument("--canonical-db", default="out/config/tracker_history.sqlite")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--probe-interval-sec", type=int, default=10)
    parser.add_argument("--current-duration-sec", type=int, default=600)
    parser.add_argument("--clean-duration-sec", type=int, default=3600)
    parser.add_argument("--current-checkpoints-sec", default="300,600")
    parser.add_argument("--clean-checkpoints-sec", default="1800,3600")
    parser.add_argument("--diagnostic-duration-sec", type=int, default=60)
    parser.add_argument("--max-symbols", type=int, default=0)
    parser.add_argument("--symbol-discovery-enabled", type=int, default=1)
    parser.add_argument("--skip-baseline-tests", action="store_true")
    parser.add_argument("--skip-reset-smoke", action="store_true")
    args = parser.parse_args()

    output_dir, canonical_db = resolve_certification_paths(args.output_dir, args.canonical_db)
    output_dir.mkdir(parents=True, exist_ok=True)
    current_clone_db = output_dir / "tracks" / "current_clone" / "tracker_history.sqlite"
    clean_cycle_db = output_dir / "tracks" / "clean_cycle" / "tracker_history.sqlite"
    reset_smoke_db = output_dir / "tracks" / "reset_smoke" / "tracker_history.sqlite"

    baseline = {"ok": True, "command": "skipped", "stdout": "", "stderr": "", "returncode": 0}
    if not args.skip_baseline_tests:
        baseline = _run_baseline_tests()
        if not baseline.get("ok"):
            raise RuntimeError("Baseline pytest run failed.")

    current_clone_files = clone_db_family(canonical_db, current_clone_db)
    current_pre = collect_sqlite_integrity(current_clone_db)
    current_runtime = run_server_track(
        track_name="current_clone",
        db_path=current_clone_db,
        output_dir=output_dir / "tracks" / "current_clone",
        duration_sec=int(args.current_duration_sec),
        checkpoint_seconds=[int(value) for value in str(args.current_checkpoints_sec).split(",") if str(value).strip()],
        base_url=args.base_url,
        probe_interval_sec=int(args.probe_interval_sec),
        max_symbols=int(args.max_symbols),
        symbol_discovery_enabled=int(args.symbol_discovery_enabled),
        diagnostic_duration_sec=int(args.diagnostic_duration_sec),
    )
    current_post = collect_sqlite_integrity(current_clone_db)

    if clean_cycle_db.exists():
        for suffix in ("", "-wal", "-shm"):
            stale = Path(f"{clean_cycle_db}{suffix}")
            if stale.exists():
                stale.unlink()
    clean_pre = collect_sqlite_integrity(clean_cycle_db)
    clean_runtime = run_server_track(
        track_name="clean_cycle",
        db_path=clean_cycle_db,
        output_dir=output_dir / "tracks" / "clean_cycle",
        duration_sec=int(args.clean_duration_sec),
        checkpoint_seconds=[int(value) for value in str(args.clean_checkpoints_sec).split(",") if str(value).strip()],
        base_url=args.base_url,
        probe_interval_sec=int(args.probe_interval_sec),
        max_symbols=int(args.max_symbols),
        symbol_discovery_enabled=int(args.symbol_discovery_enabled),
        diagnostic_duration_sec=0,
    )
    clean_post = collect_sqlite_integrity(clean_cycle_db)

    reset_smoke = {}
    if not args.skip_reset_smoke:
        clone_db_family(canonical_db, reset_smoke_db)
        reset_runtime = run_server_track(
            track_name="reset_smoke",
            db_path=reset_smoke_db,
            output_dir=output_dir / "tracks" / "reset_smoke",
            duration_sec=max(10, int(args.probe_interval_sec)),
            checkpoint_seconds=[],
            base_url=args.base_url,
            probe_interval_sec=int(args.probe_interval_sec),
            max_symbols=int(args.max_symbols),
            symbol_discovery_enabled=int(args.symbol_discovery_enabled),
            diagnostic_duration_sec=0,
            run_reset_smoke=True,
        )
        reset_smoke = reset_runtime.get("reset_smoke", {})

    current_clone_dataset_matrix = (
        collect_dataset_matrix(current_clone_db)
        if int(args.current_duration_sec) >= 300
        else []
    )
    current_clone_preflights = (
        collect_threshold_preflights(current_clone_db)
        if int(args.current_duration_sec) >= 300
        else []
    )

    certification = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "output_dir": str(output_dir),
        "canonical_db": str(canonical_db),
        "baseline": baseline,
        "tracks": {
            "current_clone": {
                "copied_files": current_clone_files,
                "pre_integrity": current_pre,
                "post_integrity": current_post,
                "dataset_matrix": current_clone_dataset_matrix,
                "preflights": current_clone_preflights,
                "runtime": current_runtime,
            },
            "clean_cycle": {
                "pre_integrity": clean_pre,
                "post_integrity": clean_post,
                "dataset_matrix": collect_dataset_matrix(clean_cycle_db),
                "preflights": collect_threshold_preflights(clean_cycle_db),
                "runtime": clean_runtime,
            },
        },
        "reset_smoke": reset_smoke,
    }
    summary_path = output_dir / "certification_summary.json"
    report_path = output_dir / "certification_report.md"
    summary_path.write_text(json.dumps(certification, indent=2, sort_keys=True), encoding="utf-8")
    report_path.write_text(build_report(certification), encoding="utf-8")
    print(json.dumps({"summary_path": str(summary_path), "report_path": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()
