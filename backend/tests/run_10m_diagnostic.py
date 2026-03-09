import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def default_report_path(workspace_root: Path | None = None) -> Path:
    base = Path(workspace_root) if workspace_root is not None else Path(os.environ.get("GEMINI_WORKSPACE_DIR", "."))
    return base / "best_lstm_model.runtime_diagnostic.md"


def resolve_runtime_paths(
    report_path_arg: str,
    db_path_arg: str,
    *,
    workspace_root: Path | None = None,
) -> tuple[Path, Path]:
    base = Path(workspace_root) if workspace_root is not None else Path(os.environ.get("GEMINI_WORKSPACE_DIR", "."))
    report_path = Path(report_path_arg) if report_path_arg else default_report_path(base)
    if not report_path.is_absolute():
        report_path = (base / report_path).resolve()
    db_path = Path(db_path_arg)
    if not db_path.is_absolute():
        db_path = (base / db_path).resolve()
    return report_path, db_path


def build_report(metrics: dict[str, float | int]) -> str:
    return f"""
# ArbML 10-Minute Runtime Diagnostic

## 1. Latency & Endpoint Health
- **Average API Response Time:** {float(metrics.get("avg_latency_ms", 0.0)):.2f} ms
- **P99 Latency:** {float(metrics.get("p99_latency_ms", 0.0)):.2f} ms
- **Max Latency:** {float(metrics.get("max_latency_ms", 0.0)):.2f} ms
- **Min Latency:** {float(metrics.get("min_latency_ms", 0.0)):.2f} ms

## 2. Tracker DB Growth
- **Initial Size:** {float(metrics.get("start_db_kb", 0.0)):.2f} KB
- **Final Size:** {float(metrics.get("end_db_kb", 0.0)):.2f} KB
- **Growth:** {float(metrics.get("db_growth_kb", 0.0)):.2f} KB

## 3. Payload Integrity
- **Total Opportunities Evaluated:** {int(metrics.get("total_samples", 0))}
- **Data Quality Errors:** {int(metrics.get("quality_errors", 0))}
- **Concurrent Tracked Pairs at End:** {int(metrics.get("records_tracked", 0))}
- **Rows Seen in Degraded State:** {int(metrics.get("degraded_rows_seen", 0))}
- **Ready Artifact Ratio:** {float(metrics.get("artifact_ready_ratio", 0.0)):.2%}

## 4. Caveat
Este diagnóstico mede saúde de runtime e integridade do payload. Ele não prova qualidade do modelo, não substitui a auditoria offline e não deve sobrescrever `ml_diagnostic_report.md`.
""".strip() + "\n"


def collect_runtime_metrics(
    *,
    api_url: str,
    db_path: Path,
    duration_sec: int = 600,
    probe_interval_sec: int = 10,
) -> dict[str, float | int]:
    iterations = max(1, int(duration_sec) // max(1, int(probe_interval_sec)))
    latencies = []
    db_sizes = []
    quality_errors = 0
    total_samples = 0
    records_tracked = 0
    degraded_rows_seen = 0
    ready_rows_seen = 0

    for index in range(iterations):
        start_t = time.time()
        try:
            req = urllib.request.urlopen(api_url, timeout=5)
            res = req.read()
            latency_ms = (time.time() - start_t) * 1000
            latencies.append(latency_ms)

            payload = json.loads(res.decode("utf-8"))
            data = payload.get("data", [])
            total_samples += len(data)

            for opp in data:
                ml = opp.get("mlContext")
                if not ml:
                    degraded_rows_seen += 1
                    continue
                if ml.get("ml_score") is None or not isinstance(ml.get("ml_score"), int):
                    quality_errors += 1
                if ml.get("signal_action") not in ["WAIT", "EXECUTE", "STRONG_EXECUTE"]:
                    quality_errors += 1
                if "success_probability" not in ml:
                    quality_errors += 1
                if ml.get("model_status") not in ["ready", "missing_artifact", "load_failed", "insufficient_history", "stale_artifact"]:
                    quality_errors += 1
                if "inversion_probability" not in ml:
                    quality_errors += 1
                if "signal_reason" not in ml:
                    quality_errors += 1
                if ml.get("model_status") == "ready":
                    ready_rows_seen += 1
                else:
                    degraded_rows_seen += 1

            records_tracked = len(data)
        except Exception as exc:
            print(f"Error fetching data at iter {index}: {exc}")

        db_sizes.append(db_path.stat().st_size / 1024 if db_path.exists() else 0.0)
        time.sleep(max(1, int(probe_interval_sec)))

    avg_latency = statistics.mean(latencies) if latencies else 0.0
    max_latency = max(latencies) if latencies else 0.0
    min_latency = min(latencies) if latencies else 0.0
    p99_latency = max_latency if len(latencies) < 2 else statistics.quantiles(latencies, n=100, method="inclusive")[98]
    start_db = db_sizes[0] if db_sizes else 0.0
    end_db = db_sizes[-1] if db_sizes else 0.0
    db_growth = end_db - start_db
    total_rows = ready_rows_seen + degraded_rows_seen
    artifact_ready_ratio = (ready_rows_seen / total_rows) if total_rows else 0.0

    return {
        "avg_latency_ms": avg_latency,
        "p99_latency_ms": p99_latency,
        "max_latency_ms": max_latency,
        "min_latency_ms": min_latency,
        "start_db_kb": start_db,
        "end_db_kb": end_db,
        "db_growth_kb": db_growth,
        "total_samples": total_samples,
        "quality_errors": quality_errors,
        "records_tracked": records_tracked,
        "degraded_rows_seen": degraded_rows_seen,
        "artifact_ready_ratio": artifact_ready_ratio,
    }


def main():
    parser = argparse.ArgumentParser(description="Run runtime payload diagnostic against the ML dashboard.")
    parser.add_argument("--duration-sec", type=int, default=600)
    parser.add_argument("--probe-interval-sec", type=int, default=10)
    parser.add_argument("--startup-wait-sec", type=int, default=15)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--db-path", default=os.environ.get("TEAM_OP_TRACKER_DB_PATH", "out/config/tracker_history.sqlite"))
    parser.add_argument("--output", default="")
    parser.add_argument("--spawn-server", action="store_true")
    args = parser.parse_args()

    print("Starting runtime diagnostic...")
    report_path, db_path = resolve_runtime_paths(
        args.output,
        args.db_path,
        workspace_root=Path(os.getcwd()),
    )
    api_url = args.base_url.rstrip("/") + "/api/v1/ml/dashboard"
    server_process = None
    if args.spawn_server:
        server_process = subprocess.Popen(
            [sys.executable, "src/server.py"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=os.getcwd(),
            env=os.environ.copy(),
        )
        print(f"Waiting {int(args.startup_wait_sec)} seconds for server startup...")
        time.sleep(max(0, int(args.startup_wait_sec)))
    try:
        metrics = collect_runtime_metrics(
            api_url=api_url,
            db_path=db_path,
            duration_sec=int(args.duration_sec),
            probe_interval_sec=int(args.probe_interval_sec),
        )
    finally:
        if server_process is not None:
            print("Test complete. Shutting down server...")
            server_process.terminate()
            server_process.wait()

    report = build_report(metrics)
    report_path.write_text(report, encoding="utf-8")
    print("Report generated successfully.")
    print("=" * 40)
    print(report)


if __name__ == "__main__":
    main()
