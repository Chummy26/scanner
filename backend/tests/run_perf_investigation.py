import argparse
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from statistics import mean
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


SCENARIOS = ("server_only", "scanner_ui", "dashboard_ui", "both_ui")
DEFAULT_ENDPOINTS = (
    ("scanner_lite_summary", "/api/spread/opportunities-lite/summary"),
    ("scanner_lite_list", "/api/spread/opportunities-lite"),
    ("ml_dashboard_summary", "/api/v1/ml/dashboard/summary"),
    ("ml_dashboard_list", "/api/v1/ml/dashboard/list?limit=250"),
    ("debug_perf", "/api/debug/perf"),
)


def resolve_tracker_db_path(output_dir: Path) -> Path:
    return (output_dir / "tracker_history.sqlite").resolve()


def browser_targets_for_scenario(base_url: str, scenario: str) -> list[tuple[str, str]]:
    targets: list[tuple[str, str]] = []
    if scenario in {"scanner_ui", "both_ui"}:
        targets.append(("scanner", f"{base_url}/dashboards/scanner"))
    if scenario in {"dashboard_ui", "both_ui"}:
        targets.append(("dashboard", f"{base_url}/dashboard"))
    return targets


def build_browser_storage_state(base_url: str, token: str) -> dict[str, Any]:
    return {
        "origins": [
            {
                "origin": base_url.rstrip("/"),
                "localStorage": [{"name": "authToken", "value": token}],
            }
        ]
    }


def issue_benchmark_auth_token(base_url: str, *, timeout_sec: int = 20) -> str:
    body = json.dumps({"username": "benchmark", "password": "benchmark"}).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/auth/login",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_sec) as response:
        payload = json.loads(response.read().decode("utf-8"))
    token = str((payload or {}).get("token") or "")
    return token


def _append_ndjson(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True, sort_keys=True))
        handle.write("\n")


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * (percentile / 100.0)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _numeric_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "min": 0.0, "max": 0.0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
    data = [float(value) for value in values]
    return {
        "count": len(data),
        "min": round(min(data), 6),
        "max": round(max(data), 6),
        "mean": round(mean(data), 6),
        "p50": round(_percentile(data, 50.0), 6),
        "p95": round(_percentile(data, 95.0), 6),
        "p99": round(_percentile(data, 99.0), 6),
    }


def _extract_updated_at(payload: Any) -> str:
    if isinstance(payload, dict):
        summary = payload.get("summary")
        if isinstance(summary, dict):
            return str(summary.get("updated_at") or "")
        return str(payload.get("updated_at") or "")
    return ""


def _timed_json_get(base_url: str, name: str, path: str, *, timeout_sec: int = 20) -> dict[str, Any]:
    url = f"{base_url}{path}"
    started = time.perf_counter()
    sample: dict[str, Any] = {
        "timestamp": time.time(),
        "endpoint": name,
        "path": path,
        "url": url,
        "ok": False,
        "status_code": 0,
        "latency_ms": 0.0,
        "payload_bytes": 0,
        "rows": 0,
        "updated_at": "",
        "error": "",
        "book_age_p95_sec": 0.0,
        "book_age_p99_sec": 0.0,
        "book_age_max_sec": 0.0,
    }
    try:
        with urllib.request.urlopen(url, timeout=timeout_sec) as response:
            body_raw = response.read()
            payload = json.loads(body_raw.decode("utf-8"))
            sample["ok"] = int(response.status) == 200
            sample["status_code"] = int(response.status)
            sample["payload_bytes"] = len(body_raw)
            sample["updated_at"] = _extract_updated_at(payload)
            if isinstance(payload, dict):
                data = payload.get("data")
                if isinstance(data, list):
                    sample["rows"] = len(data)
                    if name == "scanner_lite_list":
                        book_ages = []
                        for item in data:
                            try:
                                buy_age = float(item.get("buyBookAge") or 0.0)
                            except Exception:
                                buy_age = 0.0
                            try:
                                sell_age = float(item.get("sellBookAge") or 0.0)
                            except Exception:
                                sell_age = 0.0
                            if buy_age > 0:
                                book_ages.append(buy_age)
                            if sell_age > 0:
                                book_ages.append(sell_age)
                        if book_ages:
                            sample["book_age_p95_sec"] = round(_percentile(book_ages, 95.0), 6)
                            sample["book_age_p99_sec"] = round(_percentile(book_ages, 99.0), 6)
                            sample["book_age_max_sec"] = round(max(book_ages), 6)
                elif isinstance(payload.get("summary"), dict):
                    try:
                        sample["rows"] = int(payload["summary"].get("total") or 0)
                    except (TypeError, ValueError):
                        sample["rows"] = 0
            sample["payload"] = payload
    except urllib.error.HTTPError as exc:
        sample["status_code"] = int(exc.code)
        sample["error"] = str(exc)
    except Exception as exc:
        sample["error"] = str(exc)
    sample["latency_ms"] = round((time.perf_counter() - started) * 1000.0, 6)
    return sample


def summarize_probe_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    endpoint_names = sorted({str(sample.get("endpoint") or "") for sample in samples})
    endpoints: dict[str, Any] = {}
    staleness: dict[str, int] = {}
    for name in endpoint_names:
        subset = [sample for sample in samples if sample.get("endpoint") == name]
        latencies = [float(sample.get("latency_ms") or 0.0) for sample in subset]
        payloads = [float(sample.get("payload_bytes") or 0.0) for sample in subset if sample.get("ok")]
        endpoints[name] = {
            "count": len(subset),
            "ok_count": sum(1 for sample in subset if sample.get("ok")),
            "error_count": sum(1 for sample in subset if not sample.get("ok")),
            "latency_ms": _numeric_summary(latencies),
            "payload_bytes": _numeric_summary(payloads),
            "book_age_p95_sec": _numeric_summary([float(sample.get("book_age_p95_sec") or 0.0) for sample in subset if float(sample.get("book_age_p95_sec") or 0.0) > 0.0]),
            "book_age_p99_sec": _numeric_summary([float(sample.get("book_age_p99_sec") or 0.0) for sample in subset if float(sample.get("book_age_p99_sec") or 0.0) > 0.0]),
            "book_age_max_sec": _numeric_summary([float(sample.get("book_age_max_sec") or 0.0) for sample in subset if float(sample.get("book_age_max_sec") or 0.0) > 0.0]),
        }

        last_updated_at = ""
        current_streak = 0
        max_streak = 0
        for sample in sorted(subset, key=lambda item: float(item.get("timestamp") or 0.0)):
            if not sample.get("ok"):
                continue
            updated_at = str(sample.get("updated_at") or "")
            if not updated_at:
                continue
            if updated_at == last_updated_at:
                current_streak += 1
            else:
                current_streak = 0
            max_streak = max(max_streak, current_streak)
            last_updated_at = updated_at
        staleness[name] = max_streak

    return {"endpoints": endpoints, "staleness": staleness}


def summarize_browser_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    pages = sorted({str(sample.get("label") or "") for sample in samples})
    result: dict[str, Any] = {}
    for label in pages:
        subset = [sample for sample in samples if sample.get("label") == label]
        text_hashes = [str(sample.get("text_hash") or "") for sample in subset]
        max_same_hash = 0
        current = 0
        last_hash = ""
        for value in text_hashes:
            if value and value == last_hash:
                current += 1
            else:
                current = 0
            max_same_hash = max(max_same_hash, current)
            last_hash = value
        result[label] = {
            "samples": len(subset),
            "body_chars": _numeric_summary([float(sample.get("body_chars") or 0.0) for sample in subset]),
            "row_count": _numeric_summary([float(sample.get("row_count") or 0.0) for sample in subset]),
            "card_count": _numeric_summary([float(sample.get("card_count") or 0.0) for sample in subset]),
            "inflight_requests": _numeric_summary([float(sample.get("inflight_requests") or 0.0) for sample in subset]),
            "console_error_count": max(int(sample.get("console_error_count") or 0) for sample in subset) if subset else 0,
            "page_error_count": max(int(sample.get("page_error_count") or 0) for sample in subset) if subset else 0,
            "max_consecutive_same_text_hash": max_same_hash,
        }
    return result


def summarize_server_perf(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        return {}
    last_payload = samples[-1].get("payload")
    if not isinstance(last_payload, dict):
        return {}
    perf = last_payload.get("perf") if isinstance(last_payload.get("perf"), dict) else {}
    runtime = last_payload.get("runtime") if isinstance(last_payload.get("runtime"), dict) else {}
    hot_windows = []
    for sample in samples:
        payload = sample.get("payload")
        if not isinstance(payload, dict):
            continue
        perf_payload = payload.get("perf") if isinstance(payload.get("perf"), dict) else {}
        lag = ((perf_payload.get("event_loop_lag_ms") or {}) if isinstance(perf_payload.get("event_loop_lag_ms"), dict) else {})
        lag_p95 = float(lag.get("p95") or 0.0)
        scanner_cycle = perf_payload.get("scanner_cycle") if isinstance(perf_payload.get("scanner_cycle"), dict) else {}
        scanner_p95 = float(((scanner_cycle.get("total_ms") or {}) if isinstance(scanner_cycle.get("total_ms"), dict) else {}).get("p95") or 0.0)
        if lag_p95 >= 250.0 or scanner_p95 >= 1000.0:
            hot_windows.append(
                {
                    "timestamp": sample.get("timestamp"),
                    "event_loop_lag_p95_ms": lag_p95,
                    "scanner_cycle_p95_ms": scanner_p95,
                    "runtime": payload.get("runtime"),
                }
            )
    return {"perf": perf, "runtime": runtime, "hot_windows": hot_windows[-20:]}


def _load_runtime_audit_summary(output_dir: Path) -> dict[str, Any]:
    summary_path = output_dir / "runtime_audit" / "summary.json"
    if not summary_path.is_file():
        return {}
    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _summarize_server_log(log_path: Path) -> dict[str, Any]:
    if not log_path.is_file():
        return {}
    reconnect_counts: dict[str, int] = {}
    disconnect_counts: dict[str, int] = {}
    try:
        for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if "reconnecting" in line:
                marker = ": ["
                if marker in line:
                    name = line.split(marker, 1)[1].split("]", 1)[0].strip().upper()
                    reconnect_counts[name] = int(reconnect_counts.get(name, 0) or 0) + 1
            if "exchange_disconnected" in line:
                marker = "exchange="
                if marker in line:
                    name = line.split(marker, 1)[1].split(",", 1)[0].strip().upper()
                    disconnect_counts[name] = int(disconnect_counts.get(name, 0) or 0) + 1
    except Exception:
        return {}
    return {
        "reconnect_counts": reconnect_counts,
        "disconnect_counts": disconnect_counts,
        "reconnect_total": sum(reconnect_counts.values()),
        "disconnect_total": sum(disconnect_counts.values()),
    }


def render_report(summary: dict[str, Any]) -> str:
    scenario = summary.get("scenario", "unknown")
    lines = [
        f"# Perf Investigation: {scenario}",
        "",
        "## Summary",
        f"- Duration: `{summary.get('duration_sec', 0)}s`",
        f"- Samples: `{summary.get('sample_count', 0)}`",
        f"- Browser enabled: `{summary.get('browser_enabled', False)}`",
        "",
        "## Endpoints",
    ]
    for name, metrics in sorted((summary.get("probe_summary") or {}).get("endpoints", {}).items()):
        latency = metrics.get("latency_ms", {})
        payload = metrics.get("payload_bytes", {})
        book_age_p95 = metrics.get("book_age_p95_sec", {})
        book_age_p99 = metrics.get("book_age_p99_sec", {})
        lines.append(
            f"- `{name}`: ok `{metrics.get('ok_count', 0)}/{metrics.get('count', 0)}`, "
            f"p95 `{latency.get('p95', 0)}ms`, p99 `{latency.get('p99', 0)}ms`, "
            f"payload p95 `{payload.get('p95', 0)}B`"
        )
        if metrics.get("book_age_p95_sec", {}).get("count", 0):
            lines.append(
                f"  book age p95/p99 `{book_age_p95.get('p95', 0)}s / {book_age_p99.get('p95', 0)}s`"
            )
    lines.extend(["", "## Answers"])

    probe_summary = summary.get("probe_summary") or {}
    server_perf = summary.get("server_perf") or {}
    routes = ((server_perf.get("perf") or {}).get("routes") if isinstance(server_perf.get("perf"), dict) else {}) or {}
    dashboard_sources = ((routes.get("ml_dashboard_list") or {}).get("source_counts") if isinstance(routes.get("ml_dashboard_list"), dict) else {}) or {}
    scanner_cycle = ((server_perf.get("perf") or {}).get("scanner_cycle") if isinstance(server_perf.get("perf"), dict) else {}) or {}
    cycle_metrics = scanner_cycle if isinstance(scanner_cycle, dict) else {}
    cycle_candidates = {
        "calculate_ms": float(((cycle_metrics.get("calculate_ms") or {}) if isinstance(cycle_metrics.get("calculate_ms"), dict) else {}).get("p95") or 0.0),
        "batch_record_ms": float(((cycle_metrics.get("batch_record_ms") or {}) if isinstance(cycle_metrics.get("batch_record_ms"), dict) else {}).get("p95") or 0.0),
        "tracker_enrich_ms": float(((cycle_metrics.get("tracker_enrich_ms") or {}) if isinstance(cycle_metrics.get("tracker_enrich_ms"), dict) else {}).get("p95") or 0.0),
        "history_fetch_ms": float(((cycle_metrics.get("history_fetch_ms") or {}) if isinstance(cycle_metrics.get("history_fetch_ms"), dict) else {}).get("p95") or 0.0),
        "ml_analyze_ms": float(((cycle_metrics.get("ml_analyze_ms") or {}) if isinstance(cycle_metrics.get("ml_analyze_ms"), dict) else {}).get("p95") or 0.0),
        "lite_refresh_ms": float(((cycle_metrics.get("lite_refresh_ms") or {}) if isinstance(cycle_metrics.get("lite_refresh_ms"), dict) else {}).get("p95") or 0.0),
    }
    dominant_stage = max(cycle_candidates.items(), key=lambda item: item[1])[0] if cycle_candidates else "unknown"
    lag = ((server_perf.get("perf") or {}).get("event_loop_lag_ms") if isinstance(server_perf.get("perf"), dict) else {}) or {}
    rss = summary.get("memory_rss_mb", {})
    runtime_audit = summary.get("runtime_audit", {})
    alert_counts = runtime_audit.get("alert_counts", {})
    server_log = summary.get("server_log") or {}
    scanner_summary = (probe_summary.get("endpoints") or {}).get("scanner_lite_summary", {})
    scanner_list = (probe_summary.get("endpoints") or {}).get("scanner_lite_list", {})

    lines.append(
        f"1. Scanner trava sem navegador? `{scanner_summary.get('error_count', 0) > 0 or float((scanner_summary.get('latency_ms') or {}).get('p99', 0)) >= 1000.0}`"
    )
    lines.append(
        f"2. Dashboard leve ainda faz trabalho full por request? `{bool(dashboard_sources.get('full_rehydrate', 0))}`; sources `{dashboard_sources}`"
    )
    lines.append(
        f"3. Stall dominante nasce em `calculate_all` ou em `enrich`? estágio p95 dominante: `{dominant_stage}`"
    )
    lines.append(
        f"4. Crescimento de memória é contínuo? growth `{rss.get('growth_pct_from_first', 0)}%`"
    )
    lines.append(
        f"5. Event loop lag p95/p99: `{lag.get('p95', 0)}ms / {lag.get('p99', 0)}ms`"
    )
    reconnect_total = int(alert_counts.get("exchange_reconnected", 0) or 0)
    disconnect_total = int(alert_counts.get("exchange_disconnected", 0) or 0)
    if reconnect_total <= 0 and disconnect_total <= 0 and server_log:
        reconnect_total = int(server_log.get("reconnect_total", 0) or 0)
        disconnect_total = int(server_log.get("disconnect_total", 0) or 0)
    reconnect_suffix = f"; by_exchange={server_log.get('reconnect_counts', {})}" if server_log else ""
    lines.append(
        f"6. Reconnect/disconnect alerts: `reconnected={reconnect_total}` / `disconnected={disconnect_total}`{reconnect_suffix}"
    )
    lines.append(
        f"7. Scanner book age p95/p99 (sample-level): `{((scanner_list.get('book_age_p95_sec') or {}).get('p95', 0))}s / {((scanner_list.get('book_age_p99_sec') or {}).get('p95', 0))}s`"
    )
    return "\n".join(lines) + "\n"


def _wait_for_api(base_url: str, timeout_sec: int = 45) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/api/spread/opportunities-lite/summary", timeout=5) as response:
                if int(response.status) == 200:
                    return True
        except Exception:
            time.sleep(1)
    return False


def _browser_sampler(base_url: str, scenario: str, sample_interval_sec: int, output_path: Path, stop_event: threading.Event):
    try:
        from playwright.async_api import async_playwright
    except Exception as exc:  # pragma: no cover - runtime-only path
        _append_ndjson(output_path, {"timestamp": time.time(), "kind": "browser_error", "error": f"playwright import failed: {exc}"})
        return

    async def _run() -> None:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(headless=True)
            token = ""
            try:
                token = issue_benchmark_auth_token(base_url)
            except Exception as exc:  # pragma: no cover - runtime-only path
                _append_ndjson(
                    output_path,
                    {"timestamp": time.time(), "kind": "browser_auth_error", "error": str(exc)},
                )
            context = await browser.new_context(
                storage_state=build_browser_storage_state(base_url, token) if token else None
            )
            pages: list[tuple[str, Any, dict[str, int]]] = []

            async def _ensure_logged_in(page, target_url: str) -> None:
                if "/login" not in page.url:
                    return
                await page.evaluate(
                    """async () => {
                        const response = await fetch("/auth/login", {
                          method: "POST",
                          headers: { "Content-Type": "application/json" },
                          body: JSON.stringify({ username: "benchmark", password: "benchmark" }),
                        });
                        const payload = await response.json();
                        window.localStorage.setItem("authToken", String(payload.token || ""));
                    }"""
                )
                if page.url != target_url:
                    await page.goto(target_url, wait_until="networkidle", timeout=60000)

            try:
                if token:
                    bootstrap_page = await context.new_page()
                    await bootstrap_page.goto(f"{base_url}/login", wait_until="domcontentloaded", timeout=60000)
                    await bootstrap_page.evaluate(
                        """(authToken) => {
                            window.localStorage.setItem("authToken", authToken);
                        }""",
                        token,
                    )
                    await bootstrap_page.close()
                targets = browser_targets_for_scenario(base_url, scenario)
                for label, url in targets:
                    page = await context.new_page()
                    counters = {"inflight": 0, "console_errors": 0, "page_errors": 0}

                    def _on_request(_request, _counters=counters):
                        _counters["inflight"] += 1

                    def _on_request_done(_request, _counters=counters):
                        _counters["inflight"] = max(0, _counters["inflight"] - 1)

                    def _on_console(msg, _counters=counters):
                        if msg.type == "error":
                            _counters["console_errors"] += 1

                    def _on_page_error(_error, _counters=counters):
                        _counters["page_errors"] += 1

                    page.on("request", _on_request)
                    page.on("requestfinished", _on_request_done)
                    page.on("requestfailed", _on_request_done)
                    page.on("console", _on_console)
                    page.on("pageerror", _on_page_error)
                    await page.goto(url, wait_until="networkidle", timeout=60000)
                    await _ensure_logged_in(page, url)
                    pages.append((label, page, counters))

                while not stop_event.is_set():
                    for label, page, counters in pages:
                        sample = await page.evaluate(
                            """() => {
                                const bodyText = document.body ? document.body.innerText.slice(0, 8000) : "";
                                let hash = 0;
                                for (let i = 0; i < bodyText.length; i += 1) {
                                  hash = ((hash << 5) - hash) + bodyText.charCodeAt(i);
                                  hash |= 0;
                                }
                                return {
                                  title: document.title || "",
                                  body_chars: bodyText.length,
                                  text_hash: String(hash),
                                  row_count: document.querySelectorAll("tr, [role='row']").length,
                                  card_count: document.querySelectorAll("[data-signal-card], [data-training-session-card], article, section").length,
                                };
                            }"""
                        )
                        _append_ndjson(
                            output_path,
                            {
                                "timestamp": time.time(),
                                "label": label,
                                "url": page.url,
                                "title": sample.get("title", ""),
                                "body_chars": int(sample.get("body_chars", 0) or 0),
                                "text_hash": str(sample.get("text_hash", "") or ""),
                                "row_count": int(sample.get("row_count", 0) or 0),
                                "card_count": int(sample.get("card_count", 0) or 0),
                                "inflight_requests": counters["inflight"],
                                "console_error_count": counters["console_errors"],
                                "page_error_count": counters["page_errors"],
                            },
                        )
                    stop_event.wait(timeout=sample_interval_sec)
            finally:
                await context.close()
                await browser.close()

    import asyncio

    asyncio.run(_run())


def run_single_scenario(
    *,
    scenario: str,
    duration_sec: int,
    base_url: str,
    output_dir: Path,
    probe_interval_sec: int,
    browser_sample_sec: int,
    spawn_server: bool,
    max_symbols: int,
    symbol_discovery_enabled: str,
    host: str,
    port: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = output_dir / "samples.ndjson"
    server_perf_path = output_dir / "server_perf.ndjson"
    browser_perf_path = output_dir / "browser_perf.ndjson"
    log_path = output_dir / "server.log"

    server_process = None
    env = os.environ.copy()
    env["TEAM_OP_PORT"] = str(port)
    env["TEAM_OP_BIND"] = host
    env["TEAM_OP_MAX_SYMBOLS"] = str(int(max_symbols))
    env["TEAM_OP_SYMBOL_DISCOVERY_ENABLED"] = str(symbol_discovery_enabled)
    env["TEAM_OP_TRACKER_DB_PATH"] = str(resolve_tracker_db_path(output_dir))
    env["TEAM_OP_RUNTIME_AUDIT_DIR"] = str((output_dir / "runtime_audit").resolve())
    env["TEAM_OP_RUNTIME_AUDIT_DURATION_SEC"] = str(int(duration_sec))

    if spawn_server:
        with log_path.open("wb") as log_handle:
            server_process = subprocess.Popen(
                [sys.executable, "src/server.py", "--host", host, "--port", str(port)],
                cwd=str(ROOT_DIR),
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
        if not _wait_for_api(base_url):
            raise RuntimeError(f"Server did not become ready for scenario {scenario}")

    browser_stop = threading.Event()
    browser_thread = None
    if scenario != "server_only":
        browser_thread = threading.Thread(
            target=_browser_sampler,
            args=(base_url, scenario, browser_sample_sec, browser_perf_path, browser_stop),
            daemon=True,
        )
        browser_thread.start()

    started = time.time()
    probe_samples: list[dict[str, Any]] = []
    browser_samples: list[dict[str, Any]] = []
    server_perf_samples: list[dict[str, Any]] = []
    try:
        while (time.time() - started) < duration_sec:
            for endpoint_name, path in DEFAULT_ENDPOINTS:
                sample = _timed_json_get(base_url, endpoint_name, path)
                sample_record = {key: value for key, value in sample.items() if key != "payload"}
                _append_ndjson(samples_path, sample_record)
                probe_samples.append(sample_record)
                if endpoint_name == "debug_perf" and sample.get("ok") and isinstance(sample.get("payload"), dict):
                    perf_sample = {
                        "timestamp": sample.get("timestamp"),
                        "payload": sample.get("payload"),
                    }
                    _append_ndjson(server_perf_path, perf_sample)
                    server_perf_samples.append(perf_sample)
            time.sleep(max(1, probe_interval_sec))
    finally:
        if browser_thread is not None:
            browser_stop.set()
            browser_thread.join(timeout=30)
            if browser_perf_path.is_file():
                with browser_perf_path.open("r", encoding="utf-8") as handle:
                    browser_samples = [json.loads(line) for line in handle if line.strip()]
        if server_process is not None:
            server_process.terminate()
            try:
                server_process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server_process.kill()

    probe_summary = summarize_probe_samples(probe_samples)
    server_perf_summary = summarize_server_perf(server_perf_samples)
    browser_summary = summarize_browser_samples(browser_samples)
    memory_values = []
    for perf_sample in server_perf_samples:
        payload = perf_sample.get("payload")
        runtime = payload.get("runtime") if isinstance(payload, dict) else {}
        cache_state = ((payload.get("perf") or {}).get("cache_state") if isinstance(payload.get("perf"), dict) else {}) or {}
        rss_mb = float(runtime.get("process_rss_mb") or cache_state.get("process_rss_mb") or 0.0)
        if rss_mb > 0:
            memory_values.append(rss_mb)
    memory_summary = _numeric_summary(memory_values)
    growth_pct = 0.0
    if len(memory_values) >= 2 and memory_values[0] > 0:
        growth_pct = round(((memory_values[-1] - memory_values[0]) / memory_values[0]) * 100.0, 2)

    summary = {
        "scenario": scenario,
        "duration_sec": duration_sec,
        "sample_count": len(probe_samples),
        "browser_enabled": scenario != "server_only",
        "probe_summary": probe_summary,
        "server_perf": server_perf_summary,
        "browser_perf": browser_summary,
        "runtime_audit": _load_runtime_audit_summary(output_dir),
        "server_log": _summarize_server_log(log_path),
        "memory_rss_mb": {
            **memory_summary,
            "growth_pct_from_first": growth_pct,
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "report.md").write_text(render_report(summary), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run controlled scanner/dashboard performance investigation benchmarks.")
    parser.add_argument("--scenario", default="server_only", choices=SCENARIOS + ("all",))
    parser.add_argument("--duration-sec", type=int, default=900)
    parser.add_argument("--probe-interval-sec", type=int, default=10)
    parser.add_argument("--browser-sample-sec", type=int, default=5)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8015)
    parser.add_argument("--base-url", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--spawn-server", action="store_true")
    parser.add_argument("--max-symbols", type=int, default=0)
    parser.add_argument("--symbol-discovery-enabled", default="1")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/") if args.base_url else f"http://{args.host}:{args.port}"
    if args.output_dir:
        root_output_dir = Path(args.output_dir)
        if not root_output_dir.is_absolute():
            root_output_dir = (ROOT_DIR / args.output_dir).resolve()
    else:
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        root_output_dir = (ROOT_DIR / "out" / "benchmarks" / f"perf_investigation_{stamp}").resolve()

    scenarios = SCENARIOS if args.scenario == "all" else (args.scenario,)
    aggregate: dict[str, Any] = {}
    for index, scenario in enumerate(scenarios):
        scenario_output = root_output_dir / scenario if len(scenarios) > 1 else root_output_dir
        scenario_port = args.port + index
        scenario_base_url = base_url if len(scenarios) == 1 else f"http://{args.host}:{scenario_port}"
        aggregate[scenario] = run_single_scenario(
            scenario=scenario,
            duration_sec=args.duration_sec,
            base_url=scenario_base_url,
            output_dir=scenario_output,
            probe_interval_sec=args.probe_interval_sec,
            browser_sample_sec=args.browser_sample_sec,
            spawn_server=args.spawn_server,
            max_symbols=args.max_symbols,
            symbol_discovery_enabled=args.symbol_discovery_enabled,
            host=args.host,
            port=scenario_port,
        )

    if len(scenarios) > 1:
        (root_output_dir / "matrix_summary.json").write_text(json.dumps(aggregate, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
