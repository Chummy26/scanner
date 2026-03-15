from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tests import _baseline_v3_common as common


def run_t0_check(
    *,
    run_dir: Path,
    state_path: Path,
    base_url: str = common.DEFAULT_BASE_URL,
) -> dict[str, object]:
    pragma_integrity = common.run_pragma_integrity_check(state_path)
    sqlite_integrity = common.collect_sqlite_integrity(state_path) if state_path.exists() else {}
    db_overview = common.collect_db_overview(state_path) if state_path.exists() else {}
    episode_v3 = common.collect_episode_v3_completeness(state_path) if state_path.exists() else {}
    positive_estimate = (
        common.collect_positive_episode_estimate(state_path, threshold_pct=common.DEFAULT_POSITIVE_EPISODE_THRESHOLD)
        if state_path.exists()
        else {}
    )
    interval_regularity = common.collect_recent_interval_regularity(state_path) if state_path.exists() else {}
    feature_contract = common.validate_feature_contract_smoke()
    server_lock = common.read_server_lock_status()
    quick_cert_dir = run_dir / "t0_quick_cert_artifacts"
    quick_cert = (
        common.run_quick_certification(
            state_file=state_path,
            artifact_dir=quick_cert_dir,
            sequence_length=common.DEFAULT_SEQUENCE_LENGTH,
            prediction_horizon_sec=common.DEFAULT_PREDICTION_HORIZON_SEC,
            max_certification_duration_sec=common.DEFAULT_CERT_TIMEOUT_SEC,
        )
        if state_path.exists()
        else {}
    )
    quick_cert_summary = common.summarize_certification(quick_cert)

    blocking_findings: list[str] = []
    non_blocking_findings: list[str] = []

    if not state_path.exists():
        blocking_findings.append("missing_state_db")
    if not bool(pragma_integrity.get("ok", False)):
        blocking_findings.append(f"sqlite_integrity_failed:{pragma_integrity.get('result', 'unknown')}")
    if not bool(feature_contract.get("ok", False)):
        blocking_findings.append("feature_contract_smoke_failed")

    anomalies = dict(sqlite_integrity.get("anomalies") or {})
    if int(anomalies.get("record_count_mismatches", 0) or 0) > 0:
        blocking_findings.append("sqlite_record_count_mismatch")
    if int(anomalies.get("range_mismatches", 0) or 0) > 0:
        blocking_findings.append("sqlite_range_mismatch")
    if int(anomalies.get("open_blocks_after_close", 0) or 0) > 0:
        blocking_findings.append("sqlite_open_blocks_after_close")

    quick_failure_reasons = [
        str(reason)
        for reason in quick_cert_summary.get("failure_reasons", [])
        if str(reason) not in common.NON_BLOCKING_T0_FAILURE_REASONS
    ]
    if str(quick_cert_summary.get("verdict", "UNKNOWN")) == "FAIL" and quick_failure_reasons:
        blocking_findings.append("quick_certification_failed:" + ",".join(sorted(quick_failure_reasons)))
    elif str(quick_cert_summary.get("verdict", "UNKNOWN")) == "FAIL":
        non_blocking_findings.append("quick_certification_runtime_stale_only")

    if bool(server_lock.get("stale", False)):
        non_blocking_findings.append("stale_server_lock_detected")

    max_ts = float(db_overview.get("max_ts", 0.0) or 0.0)
    freshness_lag_sec = max(0.0, float(time.time()) - max_ts) if max_ts > 0 else 0.0
    if freshness_lag_sec > 3_600:
        non_blocking_findings.append("runtime_stale_before_restart")

    payload = {
        "phase": "t0",
        "checked_at_utc": common.utc_now_iso(),
        "run_dir": str(run_dir),
        "state_path": str(state_path),
        "base_url": str(base_url.rstrip("/")),
        "server_lock": server_lock,
        "pragma_integrity": pragma_integrity,
        "sqlite_integrity": sqlite_integrity,
        "db_overview": db_overview,
        "episode_v3_completeness": episode_v3,
        "positive_episode_estimate": positive_estimate,
        "recent_interval_regularity": interval_regularity,
        "feature_contract": feature_contract,
        "quick_certification": quick_cert_summary,
        "quick_certification_path": str(run_dir / "t0_quick_certification.json"),
        "blocking_findings": blocking_findings,
        "non_blocking_findings": non_blocking_findings,
        "ok_to_restart": not blocking_findings,
        "freshness_lag_sec": freshness_lag_sec,
        "freshness_lag_hours": round(freshness_lag_sec / 3600.0, 6),
    }
    common.write_json(run_dir / "t0_quick_certification.json", quick_cert)
    common.write_json(run_dir / "t0_check.json", payload)
    common.update_run_context(
        run_dir,
        state_path=str(state_path),
        base_url=str(base_url.rstrip("/")),
        last_phase="t0",
        t0_ok=not blocking_findings,
        t0_checked_at_utc=payload["checked_at_utc"],
    )
    common.write_markdown(
        run_dir / "t0_check.md",
        common.markdown_sections(
            "ArbML v3 Baseline T+0 Check",
            [
                (
                    "Decision",
                    [
                        f"- ok_to_restart: `{payload['ok_to_restart']}`",
                        f"- blocking_findings: `{', '.join(blocking_findings) if blocking_findings else 'none'}`",
                        f"- non_blocking_findings: `{', '.join(non_blocking_findings) if non_blocking_findings else 'none'}`",
                    ],
                ),
                (
                    "DB",
                    [
                        f"- max_ts_utc: `{db_overview.get('max_ts_utc', '')}`",
                        f"- tracker_records: `{(db_overview.get('table_counts') or {}).get('tracker_records', 0)}`",
                        f"- tracker_pair_episodes: `{(db_overview.get('table_counts') or {}).get('tracker_pair_episodes', 0)}`",
                        f"- tracker_pairs: `{(db_overview.get('table_counts') or {}).get('tracker_pairs', 0)}`",
                    ],
                ),
                (
                    "Quality",
                    [
                        f"- gate06_completeness_rate: `{float(episode_v3.get('gate06_completeness_rate', episode_v3.get('v3_completeness_rate', 0.0)) or 0.0):.4f}`",
                        f"- peak_non_positive_rate: `{float(episode_v3.get('peak_non_positive_rate', 0.0) or 0.0):.4f}`",
                        f"- exit_zero_rate: `{float(episode_v3.get('exit_zero_rate', 0.0) or 0.0):.4f}`",
                        f"- positive_episode_rate: `{float(positive_estimate.get('positive_rate', 0.0) or 0.0):.4f}`",
                        f"- interval_p50_sec: `{float(interval_regularity.get('interval_p50_sec', 0.0) or 0.0):.4f}`",
                        f"- quick_cert_verdict: `{quick_cert_summary.get('verdict', 'UNKNOWN')}`",
                    ],
                ),
            ],
        ),
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run T+0 ArbML v3 baseline validation.")
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--db-path", default="")
    parser.add_argument("--base-url", default=common.DEFAULT_BASE_URL)
    args = parser.parse_args()

    run_dir = common.resolve_run_dir(args.run_dir)
    state_path = common.resolve_state_path(args.db_path)
    payload = run_t0_check(run_dir=run_dir, state_path=state_path, base_url=args.base_url)
    print(json.dumps(payload, indent=2, sort_keys=True, default=common._json_default))
    return 1 if payload["blocking_findings"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
