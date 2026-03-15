from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tests import _baseline_v3_common as common


def run_t4_check(
    *,
    run_dir: Path,
    state_path: Path,
    base_url: str,
    restart_ts: float,
) -> dict[str, object]:
    runtime_probe = common.collect_runtime_probe(base_url)
    quick_cert_artifacts = run_dir / "t4_quick_cert_artifacts"
    quick_cert_path = run_dir / "t4_quick_cert.json"
    preflight_path = run_dir / "t4_preflight.json"

    warnings: list[str] = []
    failures: list[str] = []

    quick_cert: dict[str, object] = {}
    quick_summary = {"verdict": "UNKNOWN", "failure_reasons": [], "warnings": []}
    preflight: dict[str, object] = {}

    if not bool(runtime_probe.get("health_ok")):
        failures.append("runtime_health_failed")
    elif not bool(runtime_probe.get("perf_ok")):
        warnings.append("perf_probe_failed")

    if not failures:
        quick_cert = common.run_quick_certification(
            state_file=state_path,
            artifact_dir=quick_cert_artifacts,
            sequence_length=common.DEFAULT_SEQUENCE_LENGTH,
            prediction_horizon_sec=common.DEFAULT_PREDICTION_HORIZON_SEC,
            max_certification_duration_sec=common.DEFAULT_CERT_TIMEOUT_SEC,
        )
        quick_summary = common.summarize_certification(quick_cert)
        if str(quick_summary.get("verdict", "UNKNOWN")) == "FAIL":
            failures.append("quick_certification_failed")
        warnings.extend(str(value) for value in quick_summary.get("warnings", []))

    if not failures:
        preflight = common.run_preflight_check(
            state_file=state_path,
            output_path=preflight_path,
            sequence_length=common.DEFAULT_SEQUENCE_LENGTH,
            prediction_horizon_sec=common.DEFAULT_PREDICTION_HORIZON_SEC,
            window_stride=5,
        )
        if not bool(preflight.get("qualifies_for_training")):
            failures.append("preflight_not_qualified")
        warnings.extend(str(value) for value in (preflight.get("warnings") or []))

    payload = {
        "phase": "t4",
        "checked_at_utc": common.utc_now_iso(),
        "run_dir": str(run_dir),
        "state_path": str(state_path),
        "base_url": str(base_url.rstrip("/")),
        "restart_ts": float(restart_ts),
        "restart_ts_utc": common.utc_now_iso(float(restart_ts)),
        "runtime_probe": runtime_probe,
        "quick_certification": quick_summary,
        "quick_certification_path": str(quick_cert_path),
        "preflight": preflight,
        "preflight_path": str(preflight_path),
        "warnings": sorted(set(warnings)),
        "failures": failures,
        "ok_to_continue": not failures,
    }

    if quick_cert:
        common.write_json(quick_cert_path, quick_cert)
    if preflight and not preflight_path.exists():
        common.write_json(preflight_path, preflight)

    common.write_markdown(
        run_dir / "t4_summary.md",
        common.markdown_sections(
            "ArbML v3 Baseline T+4 Check",
            [
                (
                    "Decision",
                    [
                        f"- ok_to_continue: `{payload['ok_to_continue']}`",
                        f"- failures: `{', '.join(payload['failures']) if payload['failures'] else 'none'}`",
                        f"- warnings: `{', '.join(payload['warnings']) if payload['warnings'] else 'none'}`",
                    ],
                ),
                (
                    "Runtime",
                    [
                        f"- health_ok: `{bool(runtime_probe.get('health_ok'))}`",
                        f"- perf_ok: `{bool(runtime_probe.get('perf_ok'))}`",
                        f"- rejection_rate_pct: `{float(runtime_probe.get('rejection_rate_pct', 0.0) or 0.0):.4f}`",
                    ],
                ),
                (
                    "Quick Certification",
                    [
                        f"- verdict: `{quick_summary.get('verdict', 'UNKNOWN')}`",
                        f"- failure_reasons: `{', '.join(quick_summary.get('failure_reasons', [])) or 'none'}`",
                        f"- warnings: `{', '.join(quick_summary.get('warnings', [])) or 'none'}`",
                    ],
                ),
                (
                    "Preflight",
                    [
                        f"- qualifies_for_training: `{bool(preflight.get('qualifies_for_training', False))}`",
                        f"- selection_mode: `{preflight.get('selection_mode', 'none')}`",
                        f"- selected_threshold: `{preflight.get('selected_threshold', 'n/a')}`",
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
        last_phase="t4",
        t4_ok=not failures,
        t4_checked_at_utc=payload["checked_at_utc"],
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run T+4 ArbML v3 baseline checkpoint.")
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--db-path", default="")
    parser.add_argument("--base-url", default=common.DEFAULT_BASE_URL)
    parser.add_argument("--restart-ts", default="")
    args = parser.parse_args()

    run_dir = common.resolve_run_dir(args.run_dir)
    state_path = common.resolve_state_path(args.db_path)
    restart_ts = common.resolve_restart_ts(run_dir, args.restart_ts)
    payload = run_t4_check(
        run_dir=run_dir,
        state_path=state_path,
        base_url=args.base_url,
        restart_ts=restart_ts,
    )
    print(json.dumps(payload, indent=2, sort_keys=True, default=common._json_default))
    return 1 if payload["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
