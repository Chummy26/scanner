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


def run_monitor(
    *,
    run_dir: Path,
    state_path: Path,
    base_url: str,
    restart_ts: float,
    duration_hours: float = common.DEFAULT_FRESH_HOURS,
    interval_minutes: int = common.DEFAULT_CHECKPOINT_MINUTES,
) -> dict[str, object]:
    checkpoint_path = run_dir / "runtime_checkpoints.ndjson"
    summary_path = run_dir / "runtime_monitor_summary.json"
    deadline = float(restart_ts) + max(1.0, float(duration_hours)) * 3600.0
    interval_sec = max(60, int(interval_minutes) * 60)
    started_at = common.utc_now_ts()
    previous_max_ts = 0.0
    consecutive_no_growth = 0
    warnings: list[str] = []
    failures: list[str] = []
    max_rss_mb = 0.0
    warmup_rss_mb = 0.0
    checkpoints = 0

    while True:
        now_ts = common.utc_now_ts()
        runtime_probe = common.collect_runtime_probe(base_url)
        lock_status = common.read_server_lock_status()
        db_overview = common.collect_db_overview(state_path)
        records_last_hour = common.count_records_since(state_path, since_ts=max(0.0, now_ts - 3600.0))
        episodes_since_restart = common.count_closed_episodes_since(state_path, since_ts=float(restart_ts))
        max_ts = float(db_overview.get("max_ts", 0.0) or 0.0)
        max_ts_advanced = bool(max_ts > previous_max_ts + 1e-9) if previous_max_ts > 0.0 else bool(max_ts > 0.0)
        if previous_max_ts > 0.0 and not max_ts_advanced:
            consecutive_no_growth += 1
        else:
            consecutive_no_growth = 0
        previous_max_ts = max(previous_max_ts, max_ts)

        current_rss_mb = max(
            float(runtime_probe.get("rss_mb", 0.0) or 0.0),
            float(common.get_process_rss_mb(common.current_server_pid(lock_status)) or 0.0),
        )
        if warmup_rss_mb <= 0.0 and current_rss_mb > 0.0:
            warmup_rss_mb = current_rss_mb
        max_rss_mb = max(max_rss_mb, current_rss_mb)

        checkpoint_payload = {
            "kind": "baseline_runtime_checkpoint",
            "ts": now_ts,
            "ts_utc": common.utc_now_iso(now_ts),
            "elapsed_hours_since_restart": round(max(0.0, now_ts - float(restart_ts)) / 3600.0, 6),
            "base_url": base_url.rstrip("/"),
            "runtime_probe": runtime_probe,
            "lock_status": lock_status,
            "db_overview": db_overview,
            "records_last_hour": int(records_last_hour),
            "episodes_closed_since_restart": int(episodes_since_restart),
            "max_ts_advanced": bool(max_ts_advanced),
            "consecutive_no_growth": int(consecutive_no_growth),
            "warmup_rss_mb": float(warmup_rss_mb),
            "current_rss_mb": float(current_rss_mb),
            "max_rss_mb": float(max_rss_mb),
        }
        common.append_ndjson(checkpoint_path, checkpoint_payload)
        checkpoints += 1

        if not bool(runtime_probe.get("health_ok")):
            failures.append("runtime_health_failed")
            break
        if not bool(runtime_probe.get("perf_ok")):
            warnings.append("perf_probe_failed")
        if int(consecutive_no_growth) >= 2:
            failures.append("db_growth_stalled")
            break
        if int(records_last_hour) <= 0:
            warnings.append("no_records_last_hour")
        if (now_ts - float(restart_ts)) >= 2 * 3600.0 and int(episodes_since_restart) <= 0:
            warnings.append("no_closed_episodes_after_2h")
        if float(runtime_probe.get("rejection_rate_pct", 0.0) or 0.0) >= 100.0:
            warnings.append("rejection_rate_100pct")
        if warmup_rss_mb > 0.0 and current_rss_mb > (warmup_rss_mb * 2.0):
            warnings.append("memory_growth_warning")

        if now_ts >= deadline:
            break
        time.sleep(interval_sec)

    finished_at = common.utc_now_ts()
    payload = {
        "phase": "monitor",
        "started_at_utc": common.utc_now_iso(started_at),
        "finished_at_utc": common.utc_now_iso(finished_at),
        "restart_ts": float(restart_ts),
        "restart_ts_utc": common.utc_now_iso(float(restart_ts)),
        "base_url": str(base_url.rstrip("/")),
        "run_dir": str(run_dir),
        "state_path": str(state_path),
        "duration_hours_target": float(duration_hours),
        "interval_minutes": int(interval_minutes),
        "checkpoint_count": int(checkpoints),
        "warnings": sorted(set(warnings)),
        "failures": failures,
        "completed_fresh_window": bool(finished_at >= deadline and not failures),
        "runtime_checkpoints_path": str(checkpoint_path),
        "warmup_rss_mb": float(warmup_rss_mb),
        "max_rss_mb": float(max_rss_mb),
        "final_max_ts_utc": common.utc_now_iso(previous_max_ts) if previous_max_ts > 0.0 else "",
    }
    common.write_json(summary_path, payload)
    common.write_markdown(
        run_dir / "runtime_monitor_summary.md",
        common.markdown_sections(
            "ArbML v3 Baseline Runtime Monitor",
            [
                (
                    "Decision",
                    [
                        f"- completed_fresh_window: `{payload['completed_fresh_window']}`",
                        f"- failures: `{', '.join(payload['failures']) if payload['failures'] else 'none'}`",
                        f"- warnings: `{', '.join(payload['warnings']) if payload['warnings'] else 'none'}`",
                    ],
                ),
                (
                    "Window",
                    [
                        f"- restart_ts_utc: `{payload['restart_ts_utc']}`",
                        f"- finished_at_utc: `{payload['finished_at_utc']}`",
                        f"- checkpoint_count: `{payload['checkpoint_count']}`",
                    ],
                ),
                (
                    "Memory",
                    [
                        f"- warmup_rss_mb: `{payload['warmup_rss_mb']:.3f}`",
                        f"- max_rss_mb: `{payload['max_rss_mb']:.3f}`",
                        f"- final_max_ts_utc: `{payload['final_max_ts_utc']}`",
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
        last_phase="monitor",
        monitor_completed=bool(payload["completed_fresh_window"]),
        monitor_failures=list(payload["failures"]),
        warmup_rss_mb=float(warmup_rss_mb),
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Monitor ArbML v3 fresh runtime during the 8h window.")
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--db-path", default="")
    parser.add_argument("--base-url", default=common.DEFAULT_BASE_URL)
    parser.add_argument("--restart-ts", default="")
    parser.add_argument("--duration-hours", type=float, default=common.DEFAULT_FRESH_HOURS)
    parser.add_argument("--interval-minutes", type=int, default=common.DEFAULT_CHECKPOINT_MINUTES)
    args = parser.parse_args()

    run_dir = common.resolve_run_dir(args.run_dir)
    state_path = common.resolve_state_path(args.db_path)
    restart_ts = common.resolve_restart_ts(run_dir, args.restart_ts)
    payload = run_monitor(
        run_dir=run_dir,
        state_path=state_path,
        base_url=args.base_url,
        restart_ts=restart_ts,
        duration_hours=args.duration_hours,
        interval_minutes=args.interval_minutes,
    )
    print(json.dumps(payload, indent=2, sort_keys=True, default=common._json_default))
    return 1 if payload["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
