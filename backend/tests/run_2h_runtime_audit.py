import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.spread.runtime_audit import default_runtime_audit_dir, finalize_runtime_audit_package, summarize_dashboard_payload


def _wait_for_api(base_url: str, timeout_sec: int = 30) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/api/v1/ml/dashboard", timeout=5) as response:
                return response.status == 200
        except Exception:
            time.sleep(1)
    return False


def _append_ndjson(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, ensure_ascii=True))
        handle.write("\n")


def _probe_dashboard(base_url: str, output_path: Path) -> None:
    started = time.perf_counter()
    payload_count = 0
    status_code = 0
    error = ""
    ok = False
    try:
        with urllib.request.urlopen(f"{base_url}/api/v1/ml/dashboard", timeout=10) as response:
            status_code = int(response.status)
            body = json.loads(response.read().decode("utf-8"))
            payload_count = len(body.get("data", [])) if isinstance(body, dict) else 0
            payload_summary = summarize_dashboard_payload(body if isinstance(body, dict) else {})
            ok = status_code == 200
    except urllib.error.HTTPError as exc:
        status_code = int(exc.code)
        error = str(exc)
        payload_summary = summarize_dashboard_payload({})
    except Exception as exc:  # pragma: no cover - runtime utility
        error = str(exc)
        payload_summary = summarize_dashboard_payload({})
    latency_ms = (time.perf_counter() - started) * 1000.0
    _append_ndjson(
        output_path,
        {
            "kind": "api_probe",
            "timestamp": time.time(),
            "latency_ms": round(latency_ms, 6),
            "status_code": status_code,
            "payload_count": payload_count,
            "ok": ok,
            "error": error,
            **payload_summary,
        },
    )


def _wait_for_api_down(base_url: str, timeout_sec: int = 30) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"{base_url.rstrip('/')}/api/v1/ml/dashboard", timeout=3)
        except Exception:
            return True
        time.sleep(1)
    return False


def _terminate_process_tree(server_process, *, base_url: str = "") -> None:
    if server_process is None:
        return
    try:
        if server_process.poll() is not None:
            if base_url:
                _wait_for_api_down(base_url, timeout_sec=5)
            return
    except Exception:
        pass
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(server_process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        try:
            server_process.wait(timeout=10)
        except Exception:
            pass
    else:
        server_process.terminate()
        try:
            server_process.wait(timeout=30)
        except Exception:
            server_process.kill()
    if base_url:
        _wait_for_api_down(base_url, timeout_sec=30)


def resolve_runtime_paths(output_dir_arg: str, db_path_arg: str) -> tuple[Path, Path]:
    if output_dir_arg:
        output_dir = Path(output_dir_arg)
        if not output_dir.is_absolute():
            output_dir = (Path.cwd() / output_dir).resolve()
    else:
        output_dir = default_runtime_audit_dir()
    db_path = Path(db_path_arg)
    if not db_path.is_absolute():
        db_path = (ROOT_DIR / db_path).resolve()
    return output_dir, db_path


def main():
    parser = argparse.ArgumentParser(description="Run a real multi-phase runtime audit package.")
    parser.add_argument("--duration-sec", type=int, default=7_200)
    parser.add_argument("--followup-sec", type=int, default=0)
    parser.add_argument("--probe-interval-sec", type=int, default=10)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--db-path", default="out/config/tracker_history.sqlite")
    parser.add_argument("--legacy-package-dir", default="")
    parser.add_argument("--spawn-server", action="store_true")
    parser.add_argument("--max-symbols", type=int, default=3)
    parser.add_argument("--symbol-discovery-enabled", default="0")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    output_dir, db_path = resolve_runtime_paths(args.output_dir, args.db_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    api_probe_path = output_dir / "api_probe.ndjson"

    server_process = None
    env = os.environ.copy()
    env["TEAM_OP_RUNTIME_AUDIT_DIR"] = str(output_dir)
    env["TEAM_OP_RUNTIME_AUDIT_DURATION_SEC"] = str(int(args.duration_sec))
    env["TEAM_OP_MAX_SYMBOLS"] = str(int(args.max_symbols))
    env["TEAM_OP_SYMBOL_DISCOVERY_ENABLED"] = str(args.symbol_discovery_enabled)
    env["TEAM_OP_TRACKER_DB_PATH"] = str(db_path)

    if args.spawn_server:
        server_process = subprocess.Popen(
            [sys.executable, "src/server.py"],
            cwd=str(ROOT_DIR),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if not _wait_for_api(base_url, timeout_sec=45):
            raise RuntimeError("Server did not become ready in time")

    run_status = "completed"
    try:
        started = time.time()
        while (time.time() - started) < float(args.duration_sec + args.followup_sec):
            _probe_dashboard(base_url, api_probe_path)
            time.sleep(max(1, int(args.probe_interval_sec)))
    except KeyboardInterrupt:
        run_status = "operator_cancelled"
    finally:
        if server_process is not None:
            _terminate_process_tree(server_process, base_url=base_url)

    result = finalize_runtime_audit_package(
        output_dir=output_dir,
        state_path=db_path,
        duration_sec=int(args.duration_sec),
        run_status=run_status,
        legacy_package_dir=Path(args.legacy_package_dir).resolve() if args.legacy_package_dir else None,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
