from __future__ import annotations

import argparse
import json
import shutil
import sys
import traceback
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tests import _baseline_v3_common as common


def _copy_if_exists(source_path: Path, target_path: Path) -> str:
    if not source_path.is_file():
        return ""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target_path)
    return str(target_path)


def run_t8_cycle(
    *,
    run_dir: Path,
    state_path: Path,
    base_url: str,
    restart_ts: float,
    fresh_hours: float = common.DEFAULT_FRESH_HOURS,
    max_certification_duration_sec: int = common.DEFAULT_CERT_TIMEOUT_SEC,
) -> dict[str, object]:
    now_ts = common.utc_now_ts()
    elapsed_hours = max(0.0, now_ts - float(restart_ts)) / 3600.0
    runtime_probe = common.collect_runtime_probe(base_url)
    quick_cert_path = run_dir / "quick_certification.json"
    preflight_path = run_dir / "threshold_preflight.json"
    summary_path = run_dir / "summary.md"
    snapshot_path = run_dir / "snapshot.sqlite"
    training_report_path = run_dir / "training_report.json"
    data_certification_path = run_dir / "data_certification.json"
    training_artifacts_dir = run_dir / "training_artifacts"

    warnings: list[str] = []
    failures: list[str] = []

    if not bool(runtime_probe.get("health_ok")):
        failures.append("runtime_health_failed")
    elif not bool(runtime_probe.get("perf_ok")):
        warnings.append("perf_probe_failed")
    if float(elapsed_hours) < float(fresh_hours):
        failures.append("fresh_runtime_window_incomplete")

    quick_cert: dict[str, object] = {}
    quick_summary = {"verdict": "UNKNOWN", "failure_reasons": [], "warnings": []}
    preflight: dict[str, object] = {}
    training_result: dict[str, object] = {}
    dataset_entry: dict[str, object] = {}
    dataset_catalog_error = ""
    training_error = ""
    copied_data_certification = ""

    if not failures:
        snapshot_path = common.create_fresh_snapshot(source_db=state_path, run_dir=run_dir, name=snapshot_path.name)
        quick_cert = common.run_quick_certification(
            state_file=snapshot_path,
            artifact_dir=run_dir / "quick_certification_artifacts",
            sequence_length=common.DEFAULT_SEQUENCE_LENGTH,
            prediction_horizon_sec=common.DEFAULT_PREDICTION_HORIZON_SEC,
            max_certification_duration_sec=common.DEFAULT_CERT_TIMEOUT_SEC,
        )
        common.write_json(quick_cert_path, quick_cert)
        quick_summary = common.summarize_certification(quick_cert)
        try:
            dataset_entry = common.register_snapshot_dataset(
                snapshot_path=snapshot_path,
                certification=quick_cert,
                label=snapshot_path.stem,
                role="baseline_snapshot",
                tags=["baseline", "snapshot", "staged"],
                bless=False,
                related_files=[quick_cert_path],
                extra_metadata={
                    "run_dir": str(run_dir),
                    "restart_ts_utc": common.utc_now_iso(float(restart_ts)),
                },
            )
        except Exception as exc:
            dataset_catalog_error = str(exc)
            warnings.append("dataset_catalog_failed")
        if str(quick_summary.get("verdict", "UNKNOWN")) not in {"PASS", "WARN"}:
            failures.append("quick_certification_failed")
        warnings.extend(str(value) for value in quick_summary.get("warnings", []))

    if not failures:
        preflight = common.run_preflight_check(
            state_file=snapshot_path,
            output_path=preflight_path,
            sequence_length=common.DEFAULT_SEQUENCE_LENGTH,
            prediction_horizon_sec=common.DEFAULT_PREDICTION_HORIZON_SEC,
            window_stride=5,
        )
        if not bool(preflight.get("qualifies_for_training")):
            failures.append("preflight_not_qualified")
        warnings.extend(str(value) for value in (preflight.get("warnings") or []))

    if not failures:
        try:
            training_result = common.run_full_training(
                state_file=snapshot_path,
                artifact_dir=training_artifacts_dir,
                sequence_length=common.DEFAULT_SEQUENCE_LENGTH,
                prediction_horizon_sec=common.DEFAULT_PREDICTION_HORIZON_SEC,
                window_stride=5,
                max_certification_duration_sec=max_certification_duration_sec,
            )
            common.write_json(training_report_path, training_result)
            copied_data_certification = _copy_if_exists(
                training_artifacts_dir / "data_certification.json",
                data_certification_path,
            )
            try:
                dataset_entry = common.register_snapshot_dataset(
                    snapshot_path=snapshot_path,
                    certification=quick_cert,
                    label=snapshot_path.stem,
                    role="baseline_snapshot",
                    tags=["baseline", "snapshot", "trained"],
                    bless=True,
                    related_files=[quick_cert_path, preflight_path, data_certification_path, training_report_path],
                    extra_metadata={
                        "run_dir": str(run_dir),
                        "training_report_path": str(training_report_path),
                        "model_status": str(training_result.get("model_status") or ""),
                    },
                )
            except Exception as exc:
                dataset_catalog_error = str(exc)
                warnings.append("dataset_catalog_failed")
        except Exception:
            training_error = traceback.format_exc()
            failures.append("training_cycle_failed")
            common.write_json(
                training_report_path,
                {
                    "phase": "t8_training_error",
                    "status": "failed",
                    "error": training_error,
                    "snapshot_path": str(snapshot_path),
                },
            )

    test_metrics = dict((training_result.get("metrics") or {}).get("test") or {})
    eta_metrics = dict((training_result.get("eta_metrics") or {}).get("test") or {})
    payload = {
        "phase": "t8",
        "checked_at_utc": common.utc_now_iso(),
        "run_dir": str(run_dir),
        "state_path": str(state_path),
        "snapshot_path": str(snapshot_path),
        "base_url": str(base_url.rstrip("/")),
        "restart_ts": float(restart_ts),
        "restart_ts_utc": common.utc_now_iso(float(restart_ts)),
        "elapsed_hours_since_restart": float(elapsed_hours),
        "fresh_hours_required": float(fresh_hours),
        "runtime_probe": runtime_probe,
        "quick_certification": quick_summary,
        "quick_certification_path": str(quick_cert_path),
        "preflight": preflight,
        "preflight_path": str(preflight_path),
        "training_result": {
            "model_status": str(training_result.get("model_status") or ""),
            "roc_auc": test_metrics.get("roc_auc"),
            "average_precision": test_metrics.get("average_precision"),
            "eta_mae": eta_metrics.get("mae"),
            "eta_rmse": eta_metrics.get("rmse"),
            "artifacts": dict(training_result.get("artifacts") or {}),
            "report_path": str(training_report_path),
            "data_certification_path": copied_data_certification,
        },
        "dataset_entry": dataset_entry,
        "dataset_catalog_error": dataset_catalog_error,
        "training_error": training_error,
        "warnings": sorted(set(warnings)),
        "failures": failures,
        "model_trained": bool(str(training_result.get("model_status") or "") == "trained" and not failures),
    }

    common.write_markdown(
        summary_path,
        common.markdown_sections(
            "ArbML v3 Baseline T+8",
            [
                (
                    "Decision",
                    [
                        f"- model_trained: `{payload['model_trained']}`",
                        f"- failures: `{', '.join(payload['failures']) if payload['failures'] else 'none'}`",
                        f"- warnings: `{', '.join(payload['warnings']) if payload['warnings'] else 'none'}`",
                    ],
                ),
                (
                    "Runtime",
                    [
                        f"- elapsed_hours_since_restart: `{payload['elapsed_hours_since_restart']:.4f}`",
                        f"- health_ok: `{bool(runtime_probe.get('health_ok'))}`",
                        f"- perf_ok: `{bool(runtime_probe.get('perf_ok'))}`",
                    ],
                ),
                (
                    "Quick Diagnosis",
                    [
                        f"- quick_cert_verdict: `{quick_summary.get('verdict', 'UNKNOWN')}`",
                        f"- quick_cert_failure_reasons: `{', '.join(quick_summary.get('failure_reasons', [])) or 'none'}`",
                        f"- preflight_qualifies_for_training: `{bool(preflight.get('qualifies_for_training', False))}`",
                        f"- selected_threshold: `{preflight.get('selected_threshold', 'n/a')}`",
                    ],
                ),
                (
                    "Training",
                    [
                        f"- model_status: `{payload['training_result']['model_status'] or 'not_started'}`",
                        f"- roc_auc: `{payload['training_result']['roc_auc']}`",
                        f"- average_precision: `{payload['training_result']['average_precision']}`",
                        f"- eta_mae: `{payload['training_result']['eta_mae']}`",
                        f"- eta_rmse: `{payload['training_result']['eta_rmse']}`",
                    ],
                ),
            ],
        ),
    )
    common.update_run_context(
        run_dir,
        restart_ts=float(restart_ts),
        base_url=str(base_url.rstrip("/")),
        state_path=str(state_path),
        snapshot_path=str(snapshot_path),
        last_phase="t8",
        t8_ok=bool(payload["model_trained"]),
        t8_checked_at_utc=payload["checked_at_utc"],
        training_artifacts_dir=str(training_artifacts_dir),
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run T+8 snapshot and training for ArbML v3 baseline.")
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--db-path", default="")
    parser.add_argument("--base-url", default=common.DEFAULT_BASE_URL)
    parser.add_argument("--restart-ts", default="")
    parser.add_argument("--fresh-hours", type=float, default=common.DEFAULT_FRESH_HOURS)
    parser.add_argument("--max-certification-duration-sec", type=int, default=common.DEFAULT_CERT_TIMEOUT_SEC)
    args = parser.parse_args()

    run_dir = common.resolve_run_dir(args.run_dir)
    state_path = common.resolve_state_path(args.db_path)
    restart_ts = common.resolve_restart_ts(run_dir, args.restart_ts)
    payload = run_t8_cycle(
        run_dir=run_dir,
        state_path=state_path,
        base_url=args.base_url,
        restart_ts=restart_ts,
        fresh_hours=args.fresh_hours,
        max_certification_duration_sec=args.max_certification_duration_sec,
    )
    print(json.dumps(payload, indent=2, sort_keys=True, default=common._json_default))
    return 1 if payload["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
