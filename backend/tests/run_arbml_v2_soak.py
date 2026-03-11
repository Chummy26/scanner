import argparse
import json
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.spread.soak_runbook import (
    append_ndjson,
    audit_snapshot_labeling,
    collect_http_checkpoint,
    collect_runtime_audit_checks,
    collect_snapshot_checks,
    collect_tracker_sql_checks,
    default_duration_for_stage,
    default_interval_for_stage,
    evaluate_stage1,
    evaluate_stage2,
    resolve_soak_paths,
    run_feature_history_harness,
    run_memory_window_harness,
    validate_latest_run_payload,
    write_soak_outputs,
)


def _safe_sleep(seconds: int) -> None:
    remaining = max(int(seconds), 0)
    while remaining > 0:
        time.sleep(min(remaining, 1))
        remaining -= 1


def _stage_output_dir(base_dir: Path, stage: str) -> Path:
    return base_dir / str(stage)


def _run_stage(
    *,
    stage: str,
    base_url: str,
    output_dir: Path,
    db_path: Path,
    artifact_root: Path,
    runtime_audit_root: Path,
    duration_sec: int,
    interval_sec: int,
) -> dict:
    checkpoints: list[dict] = []
    checkpoints_path = output_dir / "checkpoints.ndjson"
    started_at = time.time()
    deadline = started_at + float(max(duration_sec, 0))
    while True:
        checkpoint = collect_http_checkpoint(base_url)
        checkpoints.append(checkpoint)
        append_ndjson(checkpoints_path, checkpoint)
        if time.time() >= deadline:
            break
        _safe_sleep(interval_sec)

    sql_checks = collect_tracker_sql_checks(db_path)
    feature_harness = run_feature_history_harness(db_path)
    runtime_audit_checks = collect_runtime_audit_checks(runtime_audit_root)

    if stage == "stage1":
        stage_result = evaluate_stage1(
            checkpoints,
            sql_checks=sql_checks,
            feature_harness=feature_harness,
            runtime_audit_checks=runtime_audit_checks,
        )
        stage_result["stage_runtime_sec"] = int(time.time() - started_at)
        write_soak_outputs(output_dir=output_dir, stage_result=stage_result, checkpoints=checkpoints)
        return stage_result

    snapshot_checks = collect_snapshot_checks(artifact_root)
    memory_harness = run_memory_window_harness(db_path)
    latest_snapshot = dict(snapshot_checks.get("latest_snapshot") or {})
    latest_snapshot_path = Path(str(latest_snapshot.get("path") or "")) if latest_snapshot.get("path") else None
    labeling_audit = (
        audit_snapshot_labeling(latest_snapshot_path)
        if latest_snapshot_path is not None and latest_snapshot_path.is_file()
        else {"ok": False, "error": "latest snapshot not available"}
    )
    latest_payload = checkpoints[-1].get("http", {}).get("latest_training_run", {}).get("payload", {}) if checkpoints else {}
    post_retrain_checks = validate_latest_run_payload(latest_payload, scanner_checkpoint=checkpoints[-1] if checkpoints else None)
    stage_result = evaluate_stage2(
        checkpoints,
        snapshot_checks=snapshot_checks,
        memory_harness=memory_harness,
        labeling_audit=labeling_audit,
        post_retrain_checks=post_retrain_checks,
    )
    stage_result["sql_checks"] = sql_checks
    stage_result["feature_harness"] = feature_harness
    stage_result["runtime_audit_checks"] = runtime_audit_checks
    stage_result["stage_runtime_sec"] = int(time.time() - started_at)
    write_soak_outputs(output_dir=output_dir, stage_result=stage_result, checkpoints=checkpoints)
    return stage_result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ArbML v2 soak runbook.")
    parser.add_argument("--stage", choices=("stage1", "stage2", "both"), default="stage1")
    parser.add_argument("--duration-sec", type=int, default=0, help="Override stage duration.")
    parser.add_argument("--interval-sec", type=int, default=0, help="Override checkpoint interval.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--db-path", default="")
    parser.add_argument("--artifact-root", default="")
    parser.add_argument("--runtime-audit-dir", default="")
    args = parser.parse_args()

    paths = resolve_soak_paths(
        ROOT_DIR,
        output_dir_arg=args.output_dir,
        db_path_arg=args.db_path,
        artifact_root_arg=args.artifact_root,
        runtime_audit_dir_arg=args.runtime_audit_dir,
    )
    paths.output_dir.mkdir(parents=True, exist_ok=True)

    stages = ["stage1", "stage2"] if args.stage == "both" else [args.stage]
    results: dict[str, dict] = {}
    for stage in stages:
        stage_dir = _stage_output_dir(paths.output_dir, stage)
        stage_dir.mkdir(parents=True, exist_ok=True)
        duration_sec = int(args.duration_sec) if int(args.duration_sec) > 0 else default_duration_for_stage(stage)
        interval_sec = int(args.interval_sec) if int(args.interval_sec) > 0 else default_interval_for_stage(stage)
        results[stage] = _run_stage(
            stage=stage,
            base_url=args.base_url.rstrip("/"),
            output_dir=stage_dir,
            db_path=paths.db_path,
            artifact_root=paths.artifact_root,
            runtime_audit_root=paths.runtime_audit_root,
            duration_sec=duration_sec,
            interval_sec=interval_sec,
        )

    manifest_path = paths.output_dir / "manifest.json"
    manifest = {
        "base_url": args.base_url.rstrip("/"),
        "artifact_root": str(paths.artifact_root),
        "db_path": str(paths.db_path),
        "runtime_audit_root": str(paths.runtime_audit_root),
        "stages": results,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output_dir": str(paths.output_dir), "manifest_path": str(manifest_path), "stages": results}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
