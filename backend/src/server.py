"""Unified server for Team OP Scanner.

Serves the new frontend source in `novo frontend/frontend 2` and provides
REST + WebSocket APIs.

Integrates the spread engine (real-time exchange WS) and the existing
funding pipeline data.

Usage:
    python server.py [--port 8000] [--host 127.0.0.1]
"""

import argparse
import asyncio
import base64
import gzip as _gzip
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Optional

from aiohttp import web, WSMsgType

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, PUT, PATCH, DELETE, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, Authorization, X-CSRF-Token",
    "Access-Control-Max-Age": "86400",
}


@web.middleware
async def cors_middleware(request, handler):
    if request.method == "OPTIONS":
        return web.Response(status=204, headers=CORS_HEADERS)
    resp = await handler(request)
    resp.headers.update(CORS_HEADERS)
    return resp


# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SRC_DIR.parent
PROJECT_DIR = BACKEND_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from system.auth import AuthManager
from spread.perf_monitor import RuntimePerfMonitor

# ---------------------------------------------------------------------------
# ENV
# ---------------------------------------------------------------------------
ENV_FILE_NAMES = ("teamop.env", "gemini.env")


def _load_env():
    for name in ENV_FILE_NAMES:
        for base in [PROJECT_DIR, BACKEND_DIR, Path.cwd()]:
            p = base / name
            if p.is_file():
                for raw in p.read_text(encoding="utf-8").splitlines():
                    line = raw.strip()
                    if not line or line.startswith("#"):
                        continue
                    k, sep, v = line.partition("=")
                    if not sep:
                        continue
                    k = k.strip()
                    if k and k not in os.environ:
                        v = v.strip()
                        if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
                            v = v[1:-1]
                        os.environ[k] = v
                return


_load_env()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
HOST = os.environ.get("TEAM_OP_BIND", "127.0.0.1")
PORT = int(os.environ.get("TEAM_OP_PORT", "8000"))
ADMIN_USER = os.environ.get("TEAM_OP_ADMIN_USER", "admin")
ADMIN_PASS = os.environ.get("TEAM_OP_ADMIN_PASS", "admin123")

# Paths
FRONTEND_DIR = PROJECT_DIR / "novo frontend" / "frontend 2"
ASSETS_DIR = FRONTEND_DIR / "assets"
OUT_DIR = BACKEND_DIR / "out" / "prod"
CONFIG_DIR = BACKEND_DIR / "out" / "config"

# Ensure dirs
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("teamop_scanner")

# Suppress noisy asyncio SSL/ProactorBasePipeTransport warnings during WS connections.
# These are Windows-specific asyncio noise that don't affect functionality.
logging.getLogger("asyncio").setLevel(logging.CRITICAL)


# Debug log ring buffer — stores last N log entries for the debug monitor
from collections import deque as _deque

_debug_log_buffer: _deque = _deque(maxlen=100)


class _DebugLogHandler(logging.Handler):
    def emit(self, record):
        try:
            _debug_log_buffer.append({
                "ts": record.created,
                "level": record.levelname,
                "name": record.name,
                "msg": self.format(record),
            })
        except Exception:
            pass


_dlh = _DebugLogHandler()
_dlh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
logging.getLogger().addHandler(_dlh)

# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
AUTH_DB_PATH = CONFIG_DIR / "auth.db"
auth_mgr = AuthManager(str(AUTH_DB_PATH))
auth_mgr.ensure_admin_from_env(ADMIN_USER, ADMIN_PASS)

USER_PREFS_STORE: dict = {}  # user_id -> prefs dict
PREFS_FILE = CONFIG_DIR / "user_prefs.json"
if PREFS_FILE.exists():
    try:
        USER_PREFS_STORE.update(json.loads(PREFS_FILE.read_text(encoding="utf-8")))
        logging.getLogger(__name__).info(f"[Prefs] Loaded {len(USER_PREFS_STORE)} user prefs from disk")
    except Exception:
        pass


def _generate_jwt(user_id, username, is_admin=False):
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "sub": str(user_id),
        "username": username,
        "is_admin": is_admin,
        "exp": int(time.time()) + 3600 * 24 * 30,
    }
    h = base64.urlsafe_b64encode(json.dumps(header).encode()).decode().rstrip("=")
    p = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    return f"{h}.{p}.local_dev_signature"


def _decode_jwt(token: str) -> dict | None:
    if not token:
        return None
    try:
        parts = token.split(".")
        if len(parts) < 2:
            return None
        padded = parts[1] + "=" * (4 - len(parts[1]) % 4)
        return json.loads(base64.urlsafe_b64decode(padded))
    except Exception:
        return None


def _get_token(request) -> str:
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:]
    return ""


def _get_user(request) -> dict | None:
    token = _get_token(request)
    payload = _decode_jwt(token)
    if not payload:
        return None
    if payload.get("exp", 0) < time.time():
        return None
    return payload

# ---------------------------------------------------------------------------
# REST Handlers: Auth
# ---------------------------------------------------------------------------

async def handle_login(request):
    try:
        data = await request.json()
    except Exception:
        data = {}

    username = data.get("username") or data.get("email") or "user"
    token = _generate_jwt("1", username, True)

    return web.json_response({
        "token": token,
        "user": {
            "id": "1",
            "username": username,
            "name": username,
            "email": f"{username}@local",
            "roles": ["admin", "user"],
            "permissions": ["view_dashboard", "access_scanner"],
            "discordId": None,
            "telegramId": None,
            "firstLogin": False,
            "emailVerified": True,
            "emailVerifiedAt": "2024-01-01T00:00:00Z",
            "licenses": [{"type": 2, "status": "active", "endDate": "2099-12-31T23:59:59Z"}],
        },
    })


async def handle_register(request):
    try:
        data = await request.json()
    except Exception:
        data = {}

    username = data.get("username") or data.get("email") or "user"
    token = _generate_jwt("1", username, True)

    return web.json_response({
        "token": token,
        "user": {
            "id": "1",
            "username": username,
            "name": username,
            "email": f"{username}@local",
            "roles": ["admin", "user"],
            "permissions": ["view_dashboard", "access_scanner"],
            "discordId": None,
            "telegramId": None,
            "firstLogin": False,
            "emailVerified": True,
            "emailVerifiedAt": "2024-01-01T00:00:00Z",
            "licenses": [{"type": 2, "status": "active", "endDate": "2099-12-31T23:59:59Z"}],
        },
    })


async def handle_profile(request):
    user = _get_user(request)
    username = user.get("username", "user") if user else "user"

    return web.json_response({
        "id": user.get("sub", "1") if user else "1",
        "username": username,
        "name": username,
        "email": f"{username}@local",
        "roles": ["admin", "user"],
        "permissions": ["view_dashboard", "access_scanner"],
        "discordId": None,
        "telegramId": None,
        "firstLogin": False,
        "emailVerified": True,
        "emailVerifiedAt": "2024-01-01T00:00:00Z",
        "licenses": [{"type": 2, "status": "active", "endDate": "2099-12-31T23:59:59Z"}],
    })


async def handle_otp_request(request):
    return web.json_response({"message": "OTP sent"})


async def handle_otp_verify(request):
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)
    token = _generate_jwt("otp-user", data.get("email", "user"))
    return web.json_response({"token": token, "user": {"id": "otp-user", "username": data.get("email", "user")}})


# ---------------------------------------------------------------------------
# REST Handlers: Auth Admin (roles, permissions, users CRUD)
# ---------------------------------------------------------------------------

async def handle_roles_list(request):
    return web.json_response([
        {"id": "1", "name": "admin", "description": "Administrator"},
        {"id": "2", "name": "user", "description": "Regular user"},
    ])


async def handle_permissions_list(request):
    return web.json_response([
        {"id": "1", "name": "view_dashboard", "description": "View dashboard"},
        {"id": "2", "name": "access_scanner", "description": "Access scanner"},
    ])


async def handle_users_list(request):
    users = auth_mgr.list_users()
    return web.json_response([
        {
            "id": str(u["id"]),
            "username": u["username"],
            "is_admin": u.get("is_admin", False),
            "is_active": u.get("is_active", True),
        }
        for u in users
    ])


async def handle_generic_ok(request):
    return web.json_response({"ok": True})


# ---------------------------------------------------------------------------
# REST Handlers: Catalog
# ---------------------------------------------------------------------------

async def handle_catalog_coins(request):
    ws_mgr = request.app.get("ws_manager")
    if ws_mgr:
        symbols = ws_mgr.config.symbols
        coins = [{"symbol": s, "name": s, "networks": ["ERC20"]} for s in symbols]
    else:
        coins = [
            {"symbol": "BTC", "name": "Bitcoin", "networks": ["BTC", "BSC", "ETH"]},
            {"symbol": "ETH", "name": "Ethereum", "networks": ["ETH", "BSC", "ARB"]},
            {"symbol": "USDT", "name": "Tether", "networks": ["TRX", "ETH", "BSC", "SOL"]},
            {"symbol": "SOL", "name": "Solana", "networks": ["SOL", "BSC"]},
        ]
    return web.json_response(coins)


async def handle_catalog_exchanges(request):
    exchanges = [
        {"id": "1", "name": "MEXC", "slug": "mexc", "is_active": True},
        {"id": "2", "name": "BingX", "slug": "bingx", "is_active": True},
        {"id": "3", "name": "Gate.io", "slug": "gate", "is_active": True},
        {"id": "4", "name": "KuCoin", "slug": "kucoin", "is_active": True},
        {"id": "5", "name": "XT", "slug": "xt", "is_active": True},
        {"id": "6", "name": "Bitget", "slug": "bitget", "is_active": True},
    ]
    # Legacy bundle expects `{ items: [...] }`.
    return web.json_response({"items": exchanges})


async def handle_catalog_coins_names(request):
    ws_mgr = request.app.get("ws_manager")
    if ws_mgr:
        return web.json_response({"data": ws_mgr.config.symbols})
    return web.json_response({"data": ["BTC", "ETH", "SOL", "USDT"]})


async def handle_coins_crud(request):
    if request.method == "POST":
        return web.json_response({"ok": True})
    elif request.method == "DELETE":
        return web.json_response({"ok": True})
    elif request.method == "PATCH":
        return web.json_response({"ok": True})
    return web.json_response({"error": "Not found"}, status=404)


async def handle_coin_raw_data(request):
    """Return funding history + network data for the details modal.

    Frontend (useExchangePiP) expects:
      { items: [{market: "future", data: [{exchange: "mexc", data: {funding_history: [...], currency: {chains: [...]}}}]},
                {market: "spot",   data: [{exchange: "mexc", data: {currency: {chains: [...]}}}]}] }
    """
    from spread import market_data as _md

    coin_id = request.match_info.get("id", "").upper()
    ws_mgr = request.app.get("ws_manager")
    if not ws_mgr:
        return web.json_response({"symbol": coin_id, "items": []})

    exchanges = list(ws_mgr._exchange_instances.keys())
    items = []

    # Use cached funding rates (from periodic 60s fetch) to avoid extra API calls.
    # For networks, fetch on-demand (cached 5min per exchange:symbol).
    async def _get_futures(ex):
        networks = await _md.fetch_coin_networks(ex, coin_id)
        if isinstance(networks, Exception):
            networks = []
        # Build funding_history from cached rate (single current entry)
        cached_rate = _md.get_funding_rate(ex, coin_id)
        funding = []
        if cached_rate is not None:
            funding = [{"funding_time": int(time.time() * 1000), "funding_rate": str(cached_rate)}]
        if funding or networks:
            return {"exchange": ex, "data": {"funding_history": funding, "currency": {"chains": networks}}}
        return None

    async def _get_spot(ex):
        networks = await _md.fetch_coin_networks(ex, coin_id)
        if isinstance(networks, Exception):
            networks = []
        if networks:
            return {"exchange": ex, "data": {"currency": {"chains": networks}}}
        return None

    futures_results = await asyncio.gather(*[_get_futures(ex) for ex in exchanges])
    futures_data = [r for r in futures_results if r is not None]
    if futures_data:
        items.append({"market": "future", "data": futures_data})

    spot_results = await asyncio.gather(*[_get_spot(ex) for ex in exchanges])
    spot_data = [r for r in spot_results if r is not None]
    if spot_data:
        items.append({"market": "spot", "data": spot_data})

    return web.json_response({"symbol": coin_id, "items": items})


# ---------------------------------------------------------------------------
# REST Handlers: Finance
# ---------------------------------------------------------------------------

async def handle_finance_plans(request):
    return web.json_response([
        {"id": "pro", "name": "Pro Plan", "price": 99, "currency": "USDT"},
    ])


async def handle_finance_payments(request):
    return web.json_response([])


async def handle_finance_licences(request):
    return web.json_response([
        {"id": "1", "type": 2, "status": "active", "endDate": "2099-12-31T23:59:59Z"},
    ])


# ---------------------------------------------------------------------------
# REST Handlers: User Preferences
# ---------------------------------------------------------------------------

async def handle_user_preferences(request):
    user = _get_user(request)
    uid = user.get("sub", "anon") if user else "anon"

    if request.method == "GET":
        prefs = USER_PREFS_STORE.get(uid, {})
        return web.json_response({"preferences": prefs})

    if request.method == "PUT":
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON"}, status=400)
        USER_PREFS_STORE[uid] = data
        try:
            PREFS_FILE.parent.mkdir(parents=True, exist_ok=True)
            PREFS_FILE.write_text(json.dumps(USER_PREFS_STORE), encoding="utf-8")
        except Exception:
            pass
        return web.json_response({"ok": True, "preferences": data})

    return web.json_response({"error": "Method not allowed"}, status=405)


# ---------------------------------------------------------------------------
# REST Handlers: Spread Data
# ---------------------------------------------------------------------------

def _infer_payload_rows(payload) -> int:
    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, list):
            return len(data)
        summary = payload.get("summary")
        if isinstance(summary, dict):
            try:
                return int(summary.get("total") or 0)
            except (TypeError, ValueError):
                return 0
    if isinstance(payload, list):
        return len(payload)
    return 0


def _response_payload_bytes(response: web.StreamResponse) -> int:
    if isinstance(response, web.Response) and response.body is not None:
        return len(response.body)
    return 0


def _mark_perf_started(request) -> None:
    setattr(request, "_perf_started", time.perf_counter())


def _record_route_perf(request, route_name: str, started_perf: float, response: web.StreamResponse, *, source: str = "", rows: int = 0):
    perf_monitor: RuntimePerfMonitor | None = request.app.get("perf_monitor")
    if perf_monitor is None:
        return
    perf_monitor.record_route(
        route_name,
        latency_ms=(time.perf_counter() - started_perf) * 1000.0,
        status_code=int(getattr(response, "status", 200) or 200),
        payload_bytes=_response_payload_bytes(response),
        rows=int(rows),
        source=source,
        path=str(getattr(request, "path", "") or ""),
    )


def _json_response_with_perf(request, route_name: str, payload, *, source: str = "", status: int = 200, rows: int | None = None):
    response = web.json_response(payload, status=status)
    _record_route_perf(
        request,
        route_name,
        getattr(request, "_perf_started", time.perf_counter()),
        response,
        source=source,
        rows=_infer_payload_rows(payload) if rows is None else rows,
    )
    return response

async def handle_spread_opportunities(request):
    _mark_perf_started(request)
    ws_mgr = request.app.get("ws_manager")
    if not ws_mgr:
        return _json_response_with_perf(request, "scanner_legacy_list", {"data": [], "status": "no_engine"}, source="no_engine")
    return _json_response_with_perf(request, "scanner_legacy_list", {
        "data": ws_mgr.get_current_opportunities(),
        "status": ws_mgr.get_status(),
    }, source="full_rehydrate")


async def handle_spread_opportunities_lite(request):
    _mark_perf_started(request)
    ws_mgr = request.app.get("ws_manager")
    if not ws_mgr:
        return _json_response_with_perf(request, "scanner_lite_list", {"data": [], "summary": {"total": 0}, "status": "no_engine"}, source="no_engine")
    return _json_response_with_perf(request, "scanner_lite_list", {
        "data": ws_mgr.get_current_opportunities_lite(),
        "summary": ws_mgr.get_current_opportunities_lite_summary(),
        "status": ws_mgr.get_status(),
    }, source="cached_lite")


async def handle_spread_opportunities_lite_summary(request):
    _mark_perf_started(request)
    ws_mgr = request.app.get("ws_manager")
    if not ws_mgr:
        return _json_response_with_perf(request, "scanner_lite_summary", {"summary": {"total": 0}}, source="no_engine")
    return _json_response_with_perf(request, "scanner_lite_summary", {"summary": ws_mgr.get_current_opportunities_lite_summary()}, source="cached_lite")


async def handle_spread_opportunity_detail(request):
    _mark_perf_started(request)
    ws_mgr = request.app.get("ws_manager")
    if not ws_mgr:
        return _json_response_with_perf(request, "scanner_detail", {"error": "no_engine"}, status=503, source="no_engine", rows=0)
    pair_key = str(request.match_info.get("pair_key") or "").strip()
    detail = ws_mgr.get_scanner_opportunity_detail(pair_key)
    if not detail:
        return _json_response_with_perf(request, "scanner_detail", {"error": "not_found"}, status=404, source="detail_full", rows=0)
    return _json_response_with_perf(request, "scanner_detail", {"data": detail}, source="detail_full", rows=1)


async def handle_spread_status(request):
    ws_mgr = request.app.get("ws_manager")
    if not ws_mgr:
        return web.json_response({"status": "not_running"})
    return web.json_response(ws_mgr.get_status())


async def handle_debug_books(request):
    """Debug endpoint: show book state for a given symbol."""
    ws_mgr = request.app.get("ws_manager")
    if not ws_mgr:
        return web.json_response({"error": "not running"})
    symbol = (request.query.get("symbol", "") or "").upper()
    if not symbol:
        # List all symbols that have snapshots
        with ws_mgr.engine._lock:
            syms = sorted(ws_mgr.engine._snapshots.keys())
        return web.json_response({"symbols_with_books": len(syms), "symbols": syms})
    with ws_mgr.engine._lock:
        ex_data = ws_mgr.engine._snapshots.get(symbol, {})
        result = {}
        for ex, markets in ex_data.items():
            result[ex] = {}
            for mt, snap in markets.items():
                result[ex][mt] = {
                    "valid": snap.is_valid(),
                    "timestamp": snap.timestamp,
                    "age": time.time() - snap.timestamp if snap.timestamp else -1,
                    "bids_count": len(snap.bids) if snap.bids else 0,
                    "asks_count": len(snap.asks) if snap.asks else 0,
                    "best_bid": float(snap.bids[0][0]) if snap.bids else 0,
                    "best_ask": float(snap.asks[0][0]) if snap.asks else 0,
                }
    return web.json_response({"symbol": symbol, "books": result})


async def handle_debug_status(request):
    """Debug endpoint: comprehensive exchange and engine status for monitoring dashboard."""
    ws_mgr = request.app.get("ws_manager")
    now = time.time()

    if not ws_mgr:
        return web.json_response({
            "ts": now,
            "engine_running": False,
            "exchanges": [],
            "totals": {"symbols": 0, "books": 0, "opportunities": 0, "ws_clients": 0},
            "config": {},
        })

    exchanges_status = []
    total_books = 0
    total_symbols_set = set()

    for ex_name, instance in ws_mgr._exchange_instances.items():
        spot_books = 0
        futures_books = 0
        spot_symbols = set()
        futures_symbols = set()
        oldest_age = 0.0
        freshest_age = float("inf")
        has_any_book = False
        ages = []  # All book ages for distribution stats

        for key, book in instance._books.items():
            # key format: "exchange:SYMBOL:market_type"
            parts = key.split(":")
            if len(parts) < 3:
                continue
            sym = parts[1]
            mt = parts[2]
            if book._timestamp > 0:
                age = now - book._timestamp
                ages.append(age)
                has_any_book = True
                if age > oldest_age:
                    oldest_age = age
                if age < freshest_age:
                    freshest_age = age
                if mt == "spot":
                    spot_books += 1
                    spot_symbols.add(sym)
                elif mt == "futures":
                    futures_books += 1
                    futures_symbols.add(sym)
                total_symbols_set.add(sym)

        if not has_any_book:
            freshest_age = -1
            oldest_age = -1

        # Age distribution buckets
        age_dist = {"lt5": 0, "lt15": 0, "lt30": 0, "lt60": 0, "lt300": 0, "gt300": 0}
        for a in ages:
            if a < 5:
                age_dist["lt5"] += 1
            elif a < 15:
                age_dist["lt15"] += 1
            elif a < 30:
                age_dist["lt30"] += 1
            elif a < 60:
                age_dist["lt60"] += 1
            elif a < 300:
                age_dist["lt300"] += 1
            else:
                age_dist["gt300"] += 1

        avg_age = round(sum(ages) / len(ages), 1) if ages else -1
        sorted_ages = sorted(ages)
        median_age = round(sorted_ages[len(sorted_ages) // 2], 1) if sorted_ages else -1
        no_update = len(instance._books) - len(ages)  # Books that never received an update

        exchanges_status.append({
            "name": ex_name,
            "ws_running": not instance.shutdown.is_set(),
            "spot_books": spot_books,
            "futures_books": futures_books,
            "spot_symbols": len(spot_symbols),
            "futures_symbols": len(futures_symbols),
            "oldest_age": round(oldest_age, 1) if oldest_age >= 0 else -1,
            "freshest_age": round(freshest_age, 1) if freshest_age >= 0 else -1,
            "overflow_spot": len(instance._overflow_spot),
            "overflow_futures": len(instance._overflow_futures),
            "total_book_keys": len(instance._books),
            "avg_age": avg_age,
            "median_age": median_age,
            "no_update_books": no_update,
            "age_distribution": age_dist,
        })
        total_books += spot_books + futures_books

    summary = ws_mgr.engine.get_snapshot_summary()

    # Extra engine stats
    engine_extra = {}
    try:
        engine_extra["dirty_count"] = len(getattr(ws_mgr.engine, "_dirty_symbols", set()))
        engine_extra["last_stale_count"] = getattr(ws_mgr.engine, "_last_stale_count", 0)
        engine_extra["calc_cycle_ms"] = round(getattr(ws_mgr.engine, "_last_calc_ms", 0), 1)
    except Exception:
        pass

    # Tracker stats
    tracker_stats = {}
    try:
        tracker_stats.update(ws_mgr.tracker.get_storage_stats())
        tracker_stats["pairs_tracked"] = tracker_stats.get("pairs_in_memory", 0)
        tracker_stats["total_records"] = tracker_stats.get("records_total", 0)
    except Exception:
        pass

    # Market data cache age
    from spread import market_data as _md
    md_entries = len(_md._cache)
    md_oldest = 0
    if _md._cache:
        oldest_ts = min(e.get("ts", now) for e in _md._cache.values())
        md_oldest = round(now - oldest_ts, 1)

    return web.json_response({
        "ts": now,
        "engine_running": ws_mgr._running,
        "exchanges": exchanges_status,
        "totals": {
            "symbols": len(total_symbols_set),
            "books": total_books,
            "opportunities": summary.get("opportunities", 0),
            "ws_clients": len(ws_mgr._scanner_clients),
            "symbols_tracked": summary.get("symbols_tracked", 0),
            "last_calc": summary.get("last_calc", 0),
        },
        "engine": engine_extra,
        "tracker": tracker_stats,
        "market_data": {"cache_entries": md_entries, "oldest_age": md_oldest},
        "config": {
            "broadcast_interval_sec": ws_mgr.config.broadcast_interval_sec,
            "depth_limit": ws_mgr.config.depth_limit,
            "min_spread_pct": ws_mgr.config.min_spread_pct,
            "max_spread_pct": ws_mgr.config.max_spread_pct,
            "min_volume_usd": ws_mgr.config.min_volume_usd,
            "max_symbols": ws_mgr.config.max_symbols,
            "symbol_discovery_enabled": ws_mgr.config.symbol_discovery_enabled,
            "tracking_window_sec": ws_mgr.config.tracking_window_sec,
            "tracker_record_interval_sec": ws_mgr.config.tracker_record_interval_sec,
            "tracker_gap_threshold_sec": getattr(ws_mgr.config, "tracker_gap_threshold_sec", 0.0),
            "min_total_spread_pct": getattr(ws_mgr.config, "min_total_spread_pct", 1.0),
            "tracker_db_path": getattr(ws_mgr.config, "tracker_db_path", ""),
            "total_configured_symbols": len(ws_mgr.config.symbols),
            "total_configured_exchanges": len(ws_mgr.config.exchanges),
        },
    })


async def handle_debug_page(request):
    """Serve the standalone debug monitoring dashboard."""
    debug_html = SRC_DIR / "web" / "debug.html"
    if debug_html.is_file():
        html = debug_html.read_text(encoding="utf-8")
        return web.Response(text=html, content_type="text/html",
                            headers={"Cache-Control": "no-cache, no-store, must-revalidate"})
    return web.Response(text="debug.html not found", status=404)


async def handle_debug_logs(request):
    """Return last N log entries from ring buffer."""
    limit = min(int(request.query.get("limit", 50)), 100)
    logs = list(_debug_log_buffer)[-limit:]
    return web.json_response({"logs": logs})


async def handle_debug_perf(request):
    _mark_perf_started(request)
    perf_monitor: RuntimePerfMonitor | None = request.app.get("perf_monitor")
    ws_mgr = request.app.get("ws_manager")
    payload = {
        "perf": perf_monitor.snapshot() if perf_monitor is not None else {},
        "runtime": ws_mgr.get_perf_state() if ws_mgr and hasattr(ws_mgr, "get_perf_state") else {},
        "runtime_audit_enabled": bool(request.app.get("runtime_audit")),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    return _json_response_with_perf(request, "debug_perf", payload, source="perf_monitor")


async def handle_spread_history(request):
    ws_mgr = request.app.get("ws_manager")
    symbol = (request.match_info.get("symbol", "") or "").upper()
    buy_ex = _norm_exchange(request.query.get("buy_ex", "") or "")
    buy_mt = _norm_market(request.query.get("buy_mt", "") or "")
    sell_ex = _norm_exchange(request.query.get("sell_ex", "") or "")
    sell_mt = _norm_market(request.query.get("sell_mt", "") or "")

    if not ws_mgr:
        return web.json_response({"history": [], "stats": None, "opportunities": []})

    # If specific pair params given, return history for that pair
    if buy_ex and sell_ex:
        history = ws_mgr.tracker.get_history(symbol, buy_ex, buy_mt, sell_ex, sell_mt, limit=200)
        stats = ws_mgr.tracker.get_pair_stats(symbol, buy_ex, buy_mt, sell_ex, sell_mt)
        return web.json_response({"history": history, "stats": stats})

    # Otherwise return all opportunities for this symbol + stats
    opps = ws_mgr.engine.get_opportunities_for_symbol(symbol)
    opp_dicts = [o.to_scanner_dict() for o in opps]

    # Collect history from first matching pair
    all_history = []
    all_stats = None
    if opps:
        o = opps[0]
        all_history = ws_mgr.tracker.get_history(
            o.asset, o.buy_exchange, o.buy_market_type,
            o.sell_exchange, o.sell_market_type, limit=200)
        all_stats = ws_mgr.tracker.get_pair_stats(
            o.asset, o.buy_exchange, o.buy_market_type,
            o.sell_exchange, o.sell_market_type)

    return web.json_response({
        "symbol": symbol,
        "opportunities": opp_dicts,
        "history": all_history,
        "stats": all_stats,
    })


def _dashboard_model_status(opp: dict) -> str:
    flat = opp.get("modelStatus")
    if flat is not None:
        return str(flat or "missing_ml_context")
    ml_context = opp.get("mlContext")
    if not isinstance(ml_context, dict):
        return "missing_ml_context"
    return str(ml_context.get("model_status") or "missing_ml_context")


def _dashboard_signal_action(opp: dict) -> str:
    flat = opp.get("signalAction")
    if flat is not None:
        return str(flat or "NONE")
    ml_context = opp.get("mlContext")
    if not isinstance(ml_context, dict):
        return "NONE"
    return str(ml_context.get("signal_action") or "NONE")


def _dashboard_success_probability(opp: dict) -> float:
    if opp.get("successProbability") is not None:
        try:
            return float(opp.get("successProbability") or 0.0)
        except (TypeError, ValueError):
            return 0.0
    ml_context = opp.get("mlContext")
    if isinstance(ml_context, dict):
        try:
            return float(ml_context.get("success_probability") or 0.0)
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def _dashboard_ml_score(opp: dict) -> float:
    if opp.get("mlScore") is not None:
        try:
            return float(opp.get("mlScore") or 0.0)
        except (TypeError, ValueError):
            return 0.0
    ml_context = opp.get("mlContext")
    if isinstance(ml_context, dict):
        try:
            return float(ml_context.get("ml_score") or opp.get("mlScore") or 0.0)
        except (TypeError, ValueError):
            return 0.0
    try:
        return float(opp.get("mlScore") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _dashboard_inference_latency_ms(opp: dict) -> float:
    if opp.get("inferenceLatencyMs") is not None:
        try:
            return float(opp.get("inferenceLatencyMs") or 0.0)
        except (TypeError, ValueError):
            return 0.0
    ml_context = opp.get("mlContext")
    if not isinstance(ml_context, dict):
        return 0.0
    try:
        return float(ml_context.get("inference_latency_ms") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _dashboard_model_version_value(opp: dict) -> str:
    flat = opp.get("modelVersion")
    if flat is not None:
        return str(flat or "unavailable")
    ml_context = opp.get("mlContext")
    if not isinstance(ml_context, dict):
        return "unavailable"
    return str(ml_context.get("model_version") or "unavailable")


def _dashboard_drift_status(opp: dict) -> str:
    flat = opp.get("driftStatus")
    if flat is not None:
        return str(flat or "unknown")
    ml_context = opp.get("mlContext")
    if not isinstance(ml_context, dict):
        return "unknown"
    return str(ml_context.get("drift_status") or "unknown")


def _dashboard_context_strength(opp: dict) -> str:
    flat = opp.get("contextStrength")
    if flat is not None:
        return str(flat or "unknown")
    ml_context = opp.get("mlContext")
    if not isinstance(ml_context, dict):
        return "unknown"
    return str(ml_context.get("context_strength") or "unknown")


def _dashboard_range_status(opp: dict) -> str:
    flat = opp.get("rangeStatus")
    if flat is not None:
        return str(flat or "unknown")
    ml_context = opp.get("mlContext")
    if not isinstance(ml_context, dict):
        return "unknown"
    return str(ml_context.get("range_status") or "unknown")


def _dashboard_entry_position_label(opp: dict) -> str:
    flat = opp.get("entryPositionLabel")
    if flat is not None:
        return str(flat or "unknown")
    ml_context = opp.get("mlContext")
    if not isinstance(ml_context, dict):
        return "unknown"
    return str(ml_context.get("entry_position_label") or "unknown")


def _dashboard_eta_alignment_status(opp: dict) -> str:
    flat = opp.get("etaAlignmentStatus")
    if flat is not None:
        return str(flat or "unknown")
    ml_context = opp.get("mlContext")
    if not isinstance(ml_context, dict):
        return "unknown"
    return str(ml_context.get("eta_alignment_status") or "unknown")


def _dashboard_runtime_signal_reason_code(opp: dict) -> str:
    flat = opp.get("signalReasonCode")
    if flat:
        return str(flat)
    ml_context = opp.get("mlContext")
    if isinstance(ml_context, dict) and ml_context.get("signal_reason_code"):
        return str(ml_context.get("signal_reason_code"))
    return ""


def _dashboard_runtime_signal_reason_message(opp: dict) -> str:
    flat = opp.get("operatorMessage") or opp.get("signalReason")
    if flat:
        return str(flat)
    ml_context = opp.get("mlContext")
    if isinstance(ml_context, dict):
        for key in ("operator_message", "signal_reason"):
            value = ml_context.get(key)
            if value:
                return str(value)
    return ""


def _dashboard_signal_reason_code(opp: dict) -> str:
    runtime_reason = _dashboard_runtime_signal_reason_code(opp)
    if runtime_reason:
        return runtime_reason
    status = _dashboard_model_status(opp)
    if status != "ready":
        return status
    if _dashboard_range_status(opp) == "insufficient_empirical_context":
        return "insufficient_empirical_context"
    if _dashboard_entry_position_label(opp) == "above_band":
        return "entry_above_recurring_band"
    if _dashboard_entry_position_label(opp) == "below_band":
        return "entry_outside_recurring_band"
    if _dashboard_drift_status(opp) == "drifted":
        return "drift_active"
    signal_action = _dashboard_signal_action(opp)
    if signal_action == "STRONG_EXECUTE":
        return "strong_execute_ready"
    if signal_action == "EXECUTE":
        if _dashboard_eta_alignment_status(opp) == "divergent":
            return "eta_divergent"
        return "execute_ready"
    if signal_action == "WAIT":
        return "probability_below_threshold"
    return "unclassified"


def _dashboard_action_lane(opp: dict) -> str:
    status = _dashboard_model_status(opp)
    if status != "ready":
        return "blocked"
    if _dashboard_range_status(opp) == "insufficient_empirical_context":
        return "blocked"
    if _dashboard_entry_position_label(opp) in {"below_band", "above_band"}:
        return "blocked"
    if _dashboard_drift_status(opp) == "drifted":
        return "observe"
    if _dashboard_signal_action(opp) in {"EXECUTE", "STRONG_EXECUTE"}:
        return "execute_now"
    return "observe"


def _dashboard_risk_flags(opp: dict) -> list[str]:
    flags: list[str] = []
    status = _dashboard_model_status(opp)
    if status != "ready":
        flags.append(status)
    range_status = _dashboard_range_status(opp)
    if range_status == "insufficient_empirical_context":
        flags.append("insufficient_empirical_context")
    entry_position = _dashboard_entry_position_label(opp)
    if entry_position == "above_band":
        flags.append("above_band")
    elif entry_position == "below_band":
        flags.append("outside_band")
    if _dashboard_drift_status(opp) == "drifted":
        flags.append("drift")
    if _dashboard_inference_latency_ms(opp) > 50.0:
        flags.append("latency_alert")
    if _dashboard_eta_alignment_status(opp) == "divergent":
        flags.append("eta_divergent")
    return list(dict.fromkeys(flags))


def _dashboard_operator_message(opp: dict) -> str:
    runtime_message = _dashboard_runtime_signal_reason_message(opp)
    if runtime_message:
        return runtime_message
    status = _dashboard_model_status(opp)
    if status == "insufficient_history":
        return "Aguardar mais histórico estruturado antes de considerar ação."
    if status == "stale_artifact":
        return "Bundle desatualizado em relação ao código; linha bloqueada até atualizar o artefato."
    if status == "missing_artifact":
        return "Sem artefato treinado disponível; oportunidade bloqueada."
    if status == "load_failed":
        return "Falha ao carregar o bundle do modelo; revisar artefato."
    if status == "missing_ml_context":
        return "Sem contexto de ML nesta linha; revisar pipeline de inferência."
    if _dashboard_range_status(opp) == "insufficient_empirical_context":
        return "Faixa recorrente indisponível: ainda não há recorrência suficiente em blocos fechados recentes."
    if _dashboard_entry_position_label(opp) == "above_band":
        return "Spread acima da banda recorrente. Mantido bloqueado nesta versão conservadora."
    if _dashboard_entry_position_label(opp) == "below_band":
        return "Spread fora da banda recorrente de entrada. Aguardar reentrada na faixa."
    if _dashboard_drift_status(opp) == "drifted":
        return "Drift ativo: manter em observação mesmo com sinal favorável."
    signal_action = _dashboard_signal_action(opp)
    if signal_action == "STRONG_EXECUTE":
        return "Prioridade alta: contexto forte confirmado, probabilidade alta e faixa recorrente alinhada."
    if signal_action == "EXECUTE":
        if _dashboard_eta_alignment_status(opp) == "divergent":
            return "Acionável com ressalva: ETA do modelo diverge do padrão empírico recente."
        return "Acionável agora: faixa recorrente e probabilidade passaram o corte de execução."
    return "Observar: probabilidade ainda abaixo do corte operacional."


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(float(value) for value in values)
    rank = (len(ordered) - 1) * (percentile / 100.0)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _dashboard_status_rank(status: str) -> int:
    if status == "ready":
        return 0
    if status == "insufficient_history":
        return 1
    if status == "stale_artifact":
        return 1
    if status == "missing_artifact":
        return 1
    if status == "load_failed":
        return 1
    if status == "missing_ml_context":
        return 2
    return 1


def _dashboard_lane_rank(lane: str) -> int:
    if lane == "execute_now":
        return 0
    if lane == "observe":
        return 1
    if lane == "blocked":
        return 2
    return 3


def _dashboard_enrich_opportunity(opp: dict) -> dict:
    enriched = dict(opp)
    enriched["action_lane"] = _dashboard_action_lane(opp)
    enriched["signal_reason_code"] = _dashboard_signal_reason_code(opp)
    enriched["risk_flags"] = _dashboard_risk_flags(opp)
    enriched["operator_message"] = _dashboard_operator_message(opp)
    return enriched


def _dashboard_pair_key(opp: dict) -> str:
    symbol = str(opp.get("symbol") or opp.get("current") or opp.get("code") or "").upper()
    buy_from = _norm_exchange(str(opp.get("buyFrom") or opp.get("buy_exchange") or ""))
    buy_type = _norm_market(str(opp.get("buyType") or opp.get("buy_market_type") or ""))
    sell_to = _norm_exchange(str(opp.get("sellTo") or opp.get("sell_exchange") or ""))
    sell_type = _norm_market(str(opp.get("sellType") or opp.get("sell_market_type") or ""))
    return "|".join((symbol, buy_from, buy_type, sell_to, sell_type))


def _dashboard_list_item(opp: dict) -> dict:
    ml_context = opp.get("mlContext") if isinstance(opp.get("mlContext"), dict) else {}
    display_eta = opp.get("displayEtaSeconds")
    if display_eta is None:
        display_eta = ml_context.get("display_eta_seconds")
    recommended_entry = opp.get("recommendedEntryRange")
    if recommended_entry is None:
        recommended_entry = ml_context.get("recommended_entry_range") or "--"
    recommended_exit = opp.get("recommendedExitRange")
    if recommended_exit is None:
        recommended_exit = ml_context.get("recommended_exit_range") or "--"
    return {
        "pair_key": _dashboard_pair_key(opp),
        "symbol": str(opp.get("symbol") or opp.get("current") or opp.get("code") or ""),
        "buyFrom": opp.get("buyFrom") or opp.get("buy_exchange"),
        "sellTo": opp.get("sellTo") or opp.get("sell_exchange"),
        "buyType": opp.get("buyType") or opp.get("buy_market_type"),
        "sellType": opp.get("sellType") or opp.get("sell_market_type"),
        "entrySpread": opp.get("entrySpread") or opp.get("entry_spread_pct"),
        "exitSpread": opp.get("exitSpread") or opp.get("exit_spread_pct"),
        "action_lane": str(opp.get("action_lane") or _dashboard_action_lane(opp)),
        "signal_action": _dashboard_signal_action(opp),
        "signal_reason_code": str(opp.get("signal_reason_code") or _dashboard_signal_reason_code(opp)),
        "operator_message": str(opp.get("operator_message") or _dashboard_operator_message(opp)),
        "risk_flags": list(opp.get("risk_flags") or _dashboard_risk_flags(opp)),
        "model_status": _dashboard_model_status(opp),
        "success_probability": _dashboard_success_probability(opp),
        "ml_score": _dashboard_ml_score(opp),
        "model_version": _dashboard_model_version_value(opp),
        "range_status": _dashboard_range_status(opp),
        "context_strength": _dashboard_context_strength(opp),
        "entry_position_label": _dashboard_entry_position_label(opp),
        "eta_alignment_status": _dashboard_eta_alignment_status(opp),
        "display_eta_seconds": display_eta,
        "recommended_entry_range": recommended_entry,
        "recommended_exit_range": recommended_exit,
    }


def _dashboard_summary(opps_sorted: list[dict]) -> dict:
    signals: dict[str, int] = {}
    statuses: dict[str, int] = {}
    lane_counts: dict[str, int] = {"execute_now": 0, "observe": 0, "blocked": 0}
    reason_counts: dict[str, int] = {}
    ready_probs: list[float] = []
    ready_latencies: list[float] = []
    versions: dict[str, int] = {}
    drifted_ready = 0
    blocked_by_empirical_context = 0
    entry_above_recurring_band_count = 0
    eta_divergent_ready = 0
    for opp in opps_sorted:
        status = _dashboard_model_status(opp)
        signal = _dashboard_signal_action(opp)
        lane = str(opp.get("action_lane") or _dashboard_action_lane(opp))
        reason_code = str(opp.get("signal_reason_code") or _dashboard_signal_reason_code(opp))
        statuses[status] = statuses.get(status, 0) + 1
        signals[signal] = signals.get(signal, 0) + 1
        lane_counts[lane] = lane_counts.get(lane, 0) + 1
        reason_counts[reason_code] = reason_counts.get(reason_code, 0) + 1
        if status == "ready":
            ready_probs.append(_dashboard_success_probability(opp))
            ready_latencies.append(_dashboard_inference_latency_ms(opp))
            version = _dashboard_model_version_value(opp)
            versions[version] = versions.get(version, 0) + 1
            if _dashboard_drift_status(opp) == "drifted":
                drifted_ready += 1
            if reason_code == "insufficient_empirical_context":
                blocked_by_empirical_context += 1
            if reason_code == "entry_above_recurring_band":
                entry_above_recurring_band_count += 1
            if _dashboard_eta_alignment_status(opp) == "divergent":
                eta_divergent_ready += 1

    ready_count = statuses.get("ready", 0)
    degraded = len(opps_sorted) - ready_count
    if ready_count > 0 and degraded == 0 and drifted_ready == 0:
        bundle_status = "fully_ready"
    elif ready_count > 0 and drifted_ready > 0:
        bundle_status = "ready_with_drift"
    elif ready_count > 0:
        bundle_status = "mixed"
    else:
        bundle_status = "degraded"
    bundle_version = "unavailable"
    if versions:
        bundle_version = max(versions.items(), key=lambda item: (item[1], item[0]))[0]
    return {
        "total": len(opps_sorted),
        "ready": ready_count,
        "degraded": degraded,
        "signals": signals,
        "model_status": statuses,
        "lane_counts": lane_counts,
        "reason_counts": reason_counts,
        "avg_success_probability_ready": round(sum(ready_probs) / len(ready_probs), 2) if ready_probs else 0.0,
        "drifted_ready": drifted_ready,
        "model_versions": versions,
        "bundle_version": bundle_version,
        "bundle_status": bundle_status,
        "avg_inference_latency_ms_ready": round(mean(ready_latencies), 2) if ready_latencies else 0.0,
        "p99_inference_latency_ms_ready": round(_percentile(ready_latencies, 99.0), 2) if ready_latencies else 0.0,
        "blocked_by_empirical_context": blocked_by_empirical_context,
        "entry_above_recurring_band_count": entry_above_recurring_band_count,
        "eta_divergent_ready": eta_divergent_ready,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def _dashboard_all_enriched_opportunities(ws_mgr) -> tuple[list[dict], str]:
    source = "full_rehydrate"
    if hasattr(ws_mgr, "get_current_opportunities_lite"):
        opps = ws_mgr.get_current_opportunities_lite()
        source = "cached_lite"
    else:
        opps = ws_mgr.get_current_opportunities()
    enriched_opps = [_dashboard_enrich_opportunity(opp) for opp in opps]
    return sorted(
        enriched_opps,
        key=lambda opp: (
            _dashboard_lane_rank(str(opp.get("action_lane") or "blocked")),
            _dashboard_status_rank(_dashboard_model_status(opp)),
            -_dashboard_success_probability(opp),
            -_dashboard_ml_score(opp),
            str(opp.get("symbol") or ""),
        ),
    ), source


async def handle_ml_dashboard_api(request):
    _mark_perf_started(request)
    ws_mgr = request.app.get("ws_manager")
    if not ws_mgr:
        return _json_response_with_perf(request, "ml_dashboard_legacy", {"data": [], "summary": _dashboard_summary([])}, source="no_engine")

    opps_sorted, source = _dashboard_all_enriched_opportunities(ws_mgr)
    return _json_response_with_perf(request, "ml_dashboard_legacy", {"data": opps_sorted, "summary": _dashboard_summary(opps_sorted)}, source=source)


async def handle_ml_dashboard_summary_api(request):
    _mark_perf_started(request)
    ws_mgr = request.app.get("ws_manager")
    if not ws_mgr:
        return _json_response_with_perf(request, "ml_dashboard_summary", {"summary": _dashboard_summary([])}, source="no_engine")
    opps_sorted, source = _dashboard_all_enriched_opportunities(ws_mgr)
    return _json_response_with_perf(request, "ml_dashboard_summary", {"summary": _dashboard_summary(opps_sorted)}, source=source)


async def handle_ml_dashboard_list_api(request):
    _mark_perf_started(request)
    ws_mgr = request.app.get("ws_manager")
    if not ws_mgr:
        return _json_response_with_perf(request, "ml_dashboard_list", {
            "data": [],
            "summary": _dashboard_summary([]),
            "pagination": {"limit": 0, "offset": 0, "returned": 0},
        }, source="no_engine")

    lane = str(request.query.get("lane") or "").strip().lower()
    search = str(request.query.get("search") or "").strip().lower()
    try:
        limit = max(1, min(int(request.query.get("limit", "100")), 500))
    except (TypeError, ValueError):
        limit = 100
    try:
        offset = max(0, int(request.query.get("offset", "0")))
    except (TypeError, ValueError):
        offset = 0

    opps_sorted, source = _dashboard_all_enriched_opportunities(ws_mgr)
    filtered = []
    for opp in opps_sorted:
        if lane and str(opp.get("action_lane") or "").lower() != lane:
            continue
        symbol = str(opp.get("symbol") or opp.get("current") or opp.get("code") or "").lower()
        pair_key = _dashboard_pair_key(opp).lower()
        if search and search not in symbol and search not in pair_key:
            continue
        filtered.append(opp)
    sliced = filtered[offset: offset + limit]
    return _json_response_with_perf(request, "ml_dashboard_list", {
        "data": [_dashboard_list_item(opp) for opp in sliced],
        "summary": _dashboard_summary(opps_sorted),
        "pagination": {"limit": limit, "offset": offset, "returned": len(sliced)},
    }, source=source, rows=len(sliced))


async def handle_ml_dashboard_detail_api(request):
    _mark_perf_started(request)
    ws_mgr = request.app.get("ws_manager")
    if not ws_mgr:
        return _json_response_with_perf(request, "ml_dashboard_detail", {"error": "no_engine"}, status=503, source="no_engine", rows=0)
    pair_key = str(request.match_info.get("pair_key") or "").strip()
    detail = None
    if hasattr(ws_mgr, "get_scanner_opportunity_detail"):
        detail = ws_mgr.get_scanner_opportunity_detail(pair_key)
    if detail:
        enriched = _dashboard_enrich_opportunity(detail)
        return _json_response_with_perf(request, "ml_dashboard_detail", {"data": enriched}, source="detail_full", rows=1)
    opps_sorted, source = _dashboard_all_enriched_opportunities(ws_mgr)
    for opp in opps_sorted:
        if _dashboard_pair_key(opp) == pair_key:
            return _json_response_with_perf(request, "ml_dashboard_detail", {"data": opp}, source=source, rows=1)
    return _json_response_with_perf(request, "ml_dashboard_detail", {"error": "not_found"}, status=404, source=source, rows=0)

async def handle_ml_dashboard_page(request):
    path = SRC_DIR / "web" / "dashboard.html"
    if path.is_file():
        return web.Response(text=path.read_text(encoding="utf-8"), content_type="text/html")
    return web.Response(text="dashboard.html not found", status=404)


def _training_artifact_dir(run_id: int) -> Path:
    return CONFIG_DIR / "training_runs" / f"run_{int(run_id)}"


async def handle_ml_training_page(request):
    path = SRC_DIR / "web" / "training_dashboard.html"
    if path.is_file():
        return web.Response(text=path.read_text(encoding="utf-8"), content_type="text/html")
    return web.Response(text="training_dashboard.html not found", status=404)


async def handle_ml_training_sessions_api(request):
    ws_mgr = request.app.get("ws_manager")
    if not ws_mgr:
        return web.json_response({"config": {}, "sessions": [], "summary": {"total_sessions": 0}})
    include_open = str(request.query.get("include_open", "1")).strip().lower() not in {"0", "false", "no", "off"}
    return web.json_response(ws_mgr.tracker.list_training_sessions(include_open=include_open))


async def handle_ml_training_session_patch(request):
    ws_mgr = request.app.get("ws_manager")
    if not ws_mgr:
        return web.json_response({"error": "tracker unavailable"}, status=503)
    session_id = int(request.match_info["session_id"])
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    try:
        updated = ws_mgr.tracker.update_training_session(
            session_id,
            approved_for_training=payload.get("approved_for_training"),
            excluded_reason=payload.get("excluded_reason"),
            notes=payload.get("notes"),
        )
    except KeyError as exc:
        return web.json_response({"error": str(exc)}, status=404)
    except ValueError as exc:
        return web.json_response({"error": str(exc)}, status=400)
    return web.json_response(updated)


async def handle_ml_training_session_exceptions(request):
    ws_mgr = request.app.get("ws_manager")
    if not ws_mgr:
        return web.json_response({"error": "tracker unavailable"}, status=503)
    session_id = int(request.match_info["session_id"])
    try:
        return web.json_response(ws_mgr.tracker.get_session_exception_blocks(session_id))
    except KeyError as exc:
        return web.json_response({"error": str(exc)}, status=404)


async def handle_ml_training_cohort_preview(request):
    ws_mgr = request.app.get("ws_manager")
    if not ws_mgr:
        return web.json_response({"mode": "expanding_walk_forward", "sessions": [], "folds": [], "summary": {"eligible_sessions": 0}})
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    session_ids = payload.get("session_ids") if isinstance(payload, dict) else None
    try:
        normalized_session_ids = [int(value) for value in session_ids] if session_ids is not None else None
    except (TypeError, ValueError):
        return web.json_response({"error": "invalid session_ids"}, status=400)
    return web.json_response(ws_mgr.tracker.preview_training_cohorts(session_ids=normalized_session_ids))


async def handle_ml_training_blocks_api(request):
    ws_mgr = request.app.get("ws_manager")
    if not ws_mgr:
        return web.json_response({"config": {}, "sessions": [], "summary": {"total_sessions": 0, "total_blocks": 0}})
    session_id_raw = request.query.get("session_id")
    try:
        session_id = int(session_id_raw) if session_id_raw else None
    except (TypeError, ValueError):
        return web.json_response({"error": "invalid session_id"}, status=400)
    include_open = str(request.query.get("include_open", "1")).strip().lower() not in {"0", "false", "no", "off"}
    return web.json_response(ws_mgr.tracker.list_training_blocks(session_id=session_id, include_open=include_open))


async def handle_ml_training_block_patch(request):
    ws_mgr = request.app.get("ws_manager")
    if not ws_mgr:
        return web.json_response({"error": "tracker unavailable"}, status=503)
    block_id = int(request.match_info["block_id"])
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    try:
        updated = ws_mgr.tracker.update_training_block(
            block_id,
            selected_for_training=payload.get("selected_for_training"),
            notes=payload.get("notes"),
            disabled_reason=payload.get("disabled_reason"),
        )
    except KeyError as exc:
        return web.json_response({"error": str(exc)}, status=404)
    except ValueError as exc:
        return web.json_response({"error": str(exc)}, status=400)
    return web.json_response(updated)


async def handle_ml_training_block_split(request):
    ws_mgr = request.app.get("ws_manager")
    if not ws_mgr:
        return web.json_response({"error": "tracker unavailable"}, status=503)
    block_id = int(request.match_info["block_id"])
    try:
        payload = await request.json()
    except Exception:
        return web.json_response({"error": "invalid json"}, status=400)
    if "split_ts" not in payload:
        return web.json_response({"error": "split_ts is required"}, status=400)
    try:
        result = ws_mgr.tracker.split_block(block_id, float(payload["split_ts"]))
    except KeyError as exc:
        return web.json_response({"error": str(exc)}, status=404)
    except ValueError as exc:
        return web.json_response({"error": str(exc)}, status=400)
    return web.json_response(result)


async def handle_ml_training_block_merge_next(request):
    ws_mgr = request.app.get("ws_manager")
    if not ws_mgr:
        return web.json_response({"error": "tracker unavailable"}, status=503)
    block_id = int(request.match_info["block_id"])
    try:
        result = ws_mgr.tracker.merge_next_block(block_id)
    except KeyError as exc:
        return web.json_response({"error": str(exc)}, status=404)
    except ValueError as exc:
        return web.json_response({"error": str(exc)}, status=400)
    return web.json_response(result)


async def handle_ml_training_config_patch(request):
    ws_mgr = request.app.get("ws_manager")
    if not ws_mgr:
        return web.json_response({"error": "tracker unavailable"}, status=503)
    try:
        payload = await request.json()
    except Exception:
        return web.json_response({"error": "invalid json"}, status=400)
    if "gap_threshold_sec" not in payload and "min_total_spread_pct" not in payload:
        return web.json_response({"error": "gap_threshold_sec or min_total_spread_pct is required"}, status=400)
    try:
        if "gap_threshold_sec" in payload:
            ws_mgr.config.tracker_gap_threshold_sec = ws_mgr.tracker.update_gap_threshold_sec(float(payload["gap_threshold_sec"]))
        if "min_total_spread_pct" in payload:
            min_total_spread_pct = ws_mgr.tracker.update_min_total_spread_pct(float(payload["min_total_spread_pct"]))
            ws_mgr.config.min_total_spread_pct = min_total_spread_pct
            if getattr(ws_mgr, "ml_analyzer", None) is not None:
                ws_mgr.ml_analyzer.min_total_spread_pct = float(min_total_spread_pct)
    except ValueError as exc:
        return web.json_response({"error": str(exc)}, status=400)
    return web.json_response(
        {
            "gap_threshold_sec": float(ws_mgr.tracker.gap_threshold_sec),
            "min_total_spread_pct": float(ws_mgr.tracker.min_total_spread_pct),
        }
    )


async def handle_ml_training_resegment(request):
    ws_mgr = request.app.get("ws_manager")
    if not ws_mgr:
        return web.json_response({"error": "tracker unavailable"}, status=503)
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    session_ids = payload.get("session_ids")
    if session_ids is not None:
        session_ids = [int(value) for value in session_ids]
    try:
        result = ws_mgr.tracker.resegment_sessions(
            session_ids=session_ids,
            threshold_sec=float(payload["gap_threshold_sec"]) if "gap_threshold_sec" in payload else None,
        )
    except ValueError as exc:
        return web.json_response({"error": str(exc)}, status=400)
    if "gap_threshold_sec" in result:
        ws_mgr.config.tracker_gap_threshold_sec = float(result["gap_threshold_sec"])
    return web.json_response(result)


async def _execute_training_run(app: web.Application, run_id: int):
    ws_mgr = app.get("ws_manager")
    if not ws_mgr:
        return
    tracker = ws_mgr.tracker
    run_payload = tracker.get_training_run(run_id)
    session_ids = [int(session["session_id"]) for session in run_payload.get("sessions", [])]
    block_ids = [int(block["block_id"]) for block in run_payload["blocks"]]
    artifact_dir = _training_artifact_dir(run_id)
    tracker.update_training_run(
        run_id,
        status="running",
        started_at=time.time(),
        artifact_dir=str(artifact_dir),
    )
    try:
        from spread.train_model import run_training_loop

        result = await asyncio.to_thread(
            run_training_loop,
            state_file=Path(ws_mgr.config.tracker_db_path),
            artifact_dir=artifact_dir,
            sequence_length=int(run_payload["sequence_length"]),
            prediction_horizon_sec=int(run_payload["prediction_horizon_sec"]),
            min_total_spread_pct=float(getattr(ws_mgr.tracker, "min_total_spread_pct", 1.0) or 1.0),
            selected_session_ids=session_ids,
            selected_block_ids=block_ids,
        )
        tracker.update_training_run(
            run_id,
            status="completed",
            finished_at=time.time(),
            result=result,
            artifact_dir=str(artifact_dir),
            audit_path=str(result.get("artifacts", {}).get("audit_path", "")),
        )
    except Exception as exc:
        tracker.update_training_run(
            run_id,
            status="failed",
            finished_at=time.time(),
            error=str(exc),
            artifact_dir=str(artifact_dir),
        )
        logger.exception("[ML] Training run %s failed", run_id)
    finally:
        app.get("ml_training_tasks", {}).pop(run_id, None)


async def handle_ml_training_run_create(request):
    ws_mgr = request.app.get("ws_manager")
    if not ws_mgr:
        return web.json_response({"error": "tracker unavailable"}, status=503)
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    sequence_length = int(payload.get("sequence_length") or 15)
    prediction_horizon_sec = int(payload.get("prediction_horizon_sec") or 14_400)
    requested_session_ids = payload.get("session_ids")
    requested_block_ids = payload.get("block_ids")
    session_ids = [int(value) for value in requested_session_ids] if requested_session_ids else []
    block_ids = [int(value) for value in requested_block_ids] if requested_block_ids else []
    try:
        run_payload = ws_mgr.tracker.create_training_run(
            session_ids=session_ids or None,
            block_ids=block_ids or None,
            sequence_length=sequence_length,
            prediction_horizon_sec=prediction_horizon_sec,
            artifact_dir="",
        )
    except ValueError as exc:
        return web.json_response({"error": str(exc)}, status=400)
    run_id = int(run_payload["id"])
    task = asyncio.create_task(_execute_training_run(request.app, run_id))
    request.app["ml_training_tasks"][run_id] = task
    return web.json_response(run_payload, status=202)


async def handle_ml_training_run_status(request):
    ws_mgr = request.app.get("ws_manager")
    if not ws_mgr:
        return web.json_response({"error": "tracker unavailable"}, status=503)
    run_id = int(request.match_info["run_id"])
    try:
        return web.json_response(ws_mgr.tracker.get_training_run(run_id))
    except KeyError as exc:
        return web.json_response({"error": str(exc)}, status=404)


async def handle_ml_training_run_latest(request):
    ws_mgr = request.app.get("ws_manager")
    if not ws_mgr:
        return web.json_response({"error": "tracker unavailable"}, status=503)
    try:
        return web.json_response(ws_mgr.tracker.get_latest_training_run())
    except KeyError as exc:
        return web.json_response({"error": str(exc)}, status=404)


# ---------------------------------------------------------------------------
# REST Handlers: Funding (legacy pipeline data)
# ---------------------------------------------------------------------------

async def handle_funding_snapshot(request):
    snapshot_path = OUT_DIR / "latest" / "funding_snapshot.json"
    if snapshot_path.exists():
        try:
            data = json.loads(snapshot_path.read_text(encoding="utf-8"))
            return web.json_response(data)
        except Exception:
            pass
    return web.json_response({"data": [], "error": "no_data"})


# ---------------------------------------------------------------------------
# REST Handlers: Export API (ML / external consumers)
# ---------------------------------------------------------------------------

def _export_token() -> str:
    # Token is optional for local dev; set it in VPS/prod.
    return str(os.environ.get("TEAM_OP_EXPORT_TOKEN") or os.environ.get("ARBSCAN_EXPORT_TOKEN") or "").strip()


def _require_export_auth(request) -> Optional[web.Response]:
    tok = _export_token()
    if not tok:
        return None
    auth = str(request.headers.get("Authorization") or "")
    got = auth[7:].strip() if auth.startswith("Bearer ") else ""
    if got != tok:
        return web.json_response({"error": "unauthorized"}, status=401)
    return None


def _norm_market_token(v: str) -> str:
    s = str(v or "").strip().lower()
    if not s:
        return "unknown"
    if "spot" in s:
        return "spot"
    if "fut" in s or "swap" in s or "perp" in s:
        return "future"
    return s


def _parse_iso_ts_ms(ts_iso: str) -> Optional[int]:
    if not ts_iso:
        return None
    try:
        tss = ts_iso.strip().replace("Z", "+00:00")
        return int(datetime.fromisoformat(tss).timestamp() * 1000)
    except Exception:
        return None


async def handle_export_opportunities(request):
    deny = _require_export_auth(request)
    if deny is not None:
        return deny

    ws_mgr = request.app.get("ws_manager")
    now_ms = int(time.time() * 1000)
    if not ws_mgr:
        return web.json_response(
            {
                "schema_name": "scanner.export.opportunities",
                "schema_version": "1.0.0",
                "ts_ms": now_ms,
                "data": [],
                "status": "no_engine",
            }
        )

    raw = ws_mgr.get_current_opportunities() or []
    out = []
    for it in raw:
        if not isinstance(it, dict):
            continue
        asset = str(it.get("symbol") or it.get("asset") or "").strip().upper()
        buy_ex = str(it.get("buyFrom") or it.get("buy_exchange") or "").strip().lower()
        sell_ex = str(it.get("sellTo") or it.get("sell_exchange") or "").strip().lower()
        buy_mt = _norm_market_token(str(it.get("buyType") or it.get("buy_market") or it.get("buyMarket") or ""))
        sell_mt = _norm_market_token(str(it.get("sellType") or it.get("sell_market") or it.get("sellMarket") or ""))
        if not asset or not buy_ex or not sell_ex:
            continue

        entry = it.get("entrySpread")
        exit_ = it.get("exitSpread")
        try:
            entry = float(entry) if entry is not None else None
        except Exception:
            entry = None
        try:
            exit_ = float(exit_) if exit_ is not None else None
        except Exception:
            exit_ = None

        ts_iso = str(it.get("timestamp") or "").strip()
        ts_ms = _parse_iso_ts_ms(ts_iso) or now_ms

        hc = it.get("histCruzamento") if isinstance(it.get("histCruzamento"), dict) else {}
        inv_count = hc.get("inverted_count") if isinstance(hc, dict) else None
        try:
            inv_count = int(inv_count) if inv_count is not None else None
        except Exception:
            inv_count = None
        inv_counts = hc.get("inverted_counts") if isinstance(hc, dict) else None

        route_id = str(it.get("code") or f"{asset}:{buy_ex}:{buy_mt}->{sell_ex}:{sell_mt}").strip()
        arb_type = "spot_futures" if (buy_mt == "spot" and sell_mt == "future") else "futures_futures"

        out.append(
            {
                "route_id": route_id,
                "asset": asset,
                "arb_type": arb_type,
                "buy_exchange": buy_ex,
                "buy_market": buy_mt,
                "sell_exchange": sell_ex,
                "sell_market": sell_mt,
                "entry_spread_pct": entry,
                "exit_spread_pct": exit_,
                "buy_volume_24h_usdt": it.get("buyVol24"),
                "sell_volume_24h_usdt": it.get("sellVol24"),
                "buy_price": it.get("buyPrice"),
                "sell_price": it.get("sellPrice"),
                "buy_funding_rate": it.get("buyFundingRate"),
                "sell_funding_rate": it.get("sellFundingRate"),
                "buy_funding_interval": it.get("buyFundingInterval"),
                "sell_funding_interval": it.get("sellFundingInterval"),
                "buy_next_settlement_ms": it.get("buyNextSettlement"),
                "sell_next_settlement_ms": it.get("sellNextSettlement"),
                "buy_open_interest_usdt": it.get("buyOpenInterest"),
                "sell_open_interest_usdt": it.get("sellOpenInterest"),
                "buy_position_limit_usdt": it.get("buyPositionLimit"),
                "sell_position_limit_usdt": it.get("sellPositionLimit"),
                "inverted_count_24h": inv_count,
                "inverted_counts": inv_counts if isinstance(inv_counts, dict) else None,
                "ts_ms": ts_ms,
                "ts_iso": ts_iso or None,
            }
        )

    return web.json_response(
        {
            "schema_name": "scanner.export.opportunities",
            "schema_version": "1.1.0",
            "ts_ms": now_ms,
            "status": ws_mgr.get_status(),
            "data": out,
        }
    )


async def handle_export_history(request):
    deny = _require_export_auth(request)
    if deny is not None:
        return deny

    ws_mgr = request.app.get("ws_manager")
    now_ms = int(time.time() * 1000)
    if not ws_mgr:
        return web.json_response(
            {
                "schema_name": "scanner.export.history_batch",
                "schema_version": "1.0.0",
                "ts_ms": now_ms,
                "routes": [],
                "status": "no_engine",
            }
        )

    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        return web.json_response({"error": "invalid_json"}, status=400)

    routes = body.get("routes")
    if not isinstance(routes, list):
        return web.json_response({"error": "missing_routes"}, status=400)

    def _i(v: object, default: int) -> int:
        try:
            return int(v)  # type: ignore[arg-type]
        except Exception:
            return int(default)

    from_ts_ms = _i(body.get("from_ts_ms"), 0)
    to_ts_ms = _i(body.get("to_ts_ms"), now_ms)
    max_points = max(0, _i(body.get("max_points"), 5000))
    uniform_ms = max(0, _i(body.get("uniform_ms"), 0))

    out_routes = []
    for r in routes[:1000]:
        if not isinstance(r, dict):
            continue
        asset = str(r.get("asset") or r.get("symbol") or "").strip().upper()
        buy_ex = str(r.get("buy_exchange") or r.get("buyFrom") or "").strip().lower()
        sell_ex = str(r.get("sell_exchange") or r.get("sellTo") or "").strip().lower()
        buy_mt = _norm_market_token(str(r.get("buy_market") or r.get("buyMarket") or r.get("buyType") or ""))
        sell_mt = _norm_market_token(str(r.get("sell_market") or r.get("sellMarket") or r.get("sellType") or ""))
        if not asset or not buy_ex or not sell_ex:
            continue

        rid = str(r.get("route_id") or r.get("opp_id") or r.get("id") or f"{asset}:{buy_ex}:{buy_mt}->{sell_ex}:{sell_mt}").strip()

        # Pull tracker history and post-process to the requested window/sampling.
        records = ws_mgr.tracker.get_history(asset, buy_ex, buy_mt, sell_ex, sell_mt, limit=max(200, max_points or 5000))

        pts = []
        last_emit_ms: Optional[int] = None
        for rec in records:
            if not isinstance(rec, dict):
                continue
            try:
                ts_ms = int(float(rec.get("timestamp") or 0.0) * 1000.0)
            except Exception:
                continue
            if from_ts_ms and ts_ms < from_ts_ms:
                continue
            if to_ts_ms and ts_ms > to_ts_ms:
                continue
            if uniform_ms and last_emit_ms is not None and (ts_ms - last_emit_ms) < uniform_ms:
                continue
            try:
                entry = float(rec.get("entry_spread") or 0.0)
            except Exception:
                entry = 0.0
            try:
                exit_ = float(rec.get("exit_spread") or 0.0)
            except Exception:
                exit_ = 0.0
            pts.append({"ts_ms": ts_ms, "entry_spread_pct": entry, "exit_spread_pct": exit_})
            last_emit_ms = ts_ms

        if max_points and len(pts) > max_points:
            pts = pts[-max_points:]

        out_routes.append(
            {
                "route_id": rid,
                "asset": asset,
                "buy_exchange": buy_ex,
                "buy_market": buy_mt,
                "sell_exchange": sell_ex,
                "sell_market": sell_mt,
                "from_ts_ms": from_ts_ms or None,
                "to_ts_ms": to_ts_ms or None,
                "points": pts,
                "stats": ws_mgr.tracker.get_pair_stats(asset, buy_ex, buy_mt, sell_ex, sell_mt),
            }
        )

    return web.json_response(
        {
            "schema_name": "scanner.export.history_batch",
            "schema_version": "1.0.0",
            "ts_ms": now_ms,
            "routes": out_routes,
        }
    )


async def handle_export_funding_snapshot(request):
    deny = _require_export_auth(request)
    if deny is not None:
        return deny
    # Reuse legacy payload but wrap with a stable envelope.
    now_ms = int(time.time() * 1000)
    snapshot_path = OUT_DIR / "latest" / "funding_snapshot.json"
    data = None
    if snapshot_path.exists():
        try:
            data = json.loads(snapshot_path.read_text(encoding="utf-8"))
        except Exception:
            data = None
    return web.json_response(
        {
            "schema_name": "scanner.export.funding_snapshot",
            "schema_version": "1.0.0",
            "ts_ms": now_ms,
            "data": data,
        }
    )


async def handle_export_filters_snapshot(request):
    deny = _require_export_auth(request)
    if deny is not None:
        return deny
    now_ms = int(time.time() * 1000)
    # Best-effort: use saved UI settings when present.
    settings_path = OUT_DIR / "config" / "ui_settings.json"
    settings = None
    if settings_path.exists():
        try:
            settings = json.loads(settings_path.read_text(encoding="utf-8"))
        except Exception:
            settings = None
    return web.json_response(
        {
            "schema_name": "scanner.export.filters_snapshot",
            "schema_version": "1.0.0",
            "ts_ms": now_ms,
            "settings": settings,
        }
    )


async def handle_export_networks(request):
    deny = _require_export_auth(request)
    if deny is not None:
        return deny
    now_ms = int(time.time() * 1000)
    asset = str(request.query.get("asset") or "").strip().upper() or None
    return web.json_response(
        {
            "schema_name": "scanner.export.networks",
            "schema_version": "1.0.0",
            "ts_ms": now_ms,
            "asset": asset,
            "networks": [],
        }
    )


async def handle_export_funding_history(request):
    """Export bulk funding rate history (last 24 cycles) per exchange:symbol."""
    deny = _require_export_auth(request)
    if deny is not None:
        return deny

    from spread import market_data as _md

    now_ms = int(time.time() * 1000)
    symbols_param = str(request.query.get("symbols") or "").strip().upper()
    symbols = [s.strip() for s in symbols_param.split(",") if s.strip()] if symbols_param else []

    exchanges = ["mexc", "bingx", "gate", "bitget", "kucoin", "xt"]
    data = {}
    for ex in exchanges:
        ex_data = {}
        for sym in symbols:
            hist = _md.get_funding_history_bulk(ex, sym)
            if hist:
                ex_data[sym] = hist
        if ex_data:
            data[ex] = ex_data

    return web.json_response(
        {
            "schema_name": "scanner.export.funding_history",
            "schema_version": "1.0.0",
            "ts_ms": now_ms,
            "symbols": symbols,
            "data": data,
        }
    )


async def handle_export_futures_meta(request):
    """Export futures contract metadata (OI, funding interval, next settlement, limits)."""
    deny = _require_export_auth(request)
    if deny is not None:
        return deny

    from spread import market_data as _md

    now_ms = int(time.time() * 1000)
    symbols_param = str(request.query.get("symbols") or "").strip().upper()
    symbols = [s.strip() for s in symbols_param.split(",") if s.strip()] if symbols_param else []

    exchanges = ["mexc", "bingx", "gate", "bitget", "kucoin", "xt"]
    data = {}
    for ex in exchanges:
        ex_data = {}
        for sym in symbols:
            meta = _md.get_futures_meta(ex, sym)
            if meta:
                ex_data[sym] = {
                    "funding_interval_hours": meta.get("funding_interval"),
                    "next_settlement_ms": meta.get("next_settlement"),
                    "open_interest_usdt": meta.get("open_interest"),
                    "position_limit_usdt": meta.get("position_limit"),
                }
        if ex_data:
            data[ex] = ex_data

    return web.json_response(
        {
            "schema_name": "scanner.export.futures_meta",
            "schema_version": "1.0.0",
            "ts_ms": now_ms,
            "symbols": symbols,
            "data": data,
        }
    )


async def handle_export_orderbook(request):
    """Export L2 order book snapshots for requested routes (ArbML integration)."""
    deny = _require_export_auth(request)
    if deny is not None:
        return deny

    ws_mgr = request.app.get("ws_manager")
    now_ms = int(time.time() * 1000)
    if not ws_mgr:
        return web.json_response(
            {
                "schema_name": "scanner.export.orderbook",
                "schema_version": "1.0.0",
                "ts_ms": now_ms,
                "routes": [],
                "status": "no_engine",
            }
        )

    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        return web.json_response({"error": "invalid_json"}, status=400)

    routes = body.get("routes")
    if not isinstance(routes, list):
        return web.json_response({"error": "missing_routes"}, status=400)

    depth = max(1, min(20, int(body.get("depth", 10) or 10)))

    out_routes = []
    for r in routes[:200]:
        if not isinstance(r, dict):
            continue
        asset = str(r.get("asset") or r.get("symbol") or "").strip().upper()
        buy_ex = str(r.get("buy_exchange") or "").strip().lower()
        sell_ex = str(r.get("sell_exchange") or "").strip().lower()
        buy_mt = _norm_market_token(str(r.get("buy_market") or ""))
        sell_mt = _norm_market_token(str(r.get("sell_market") or ""))
        if not asset or not buy_ex or not sell_ex:
            continue

        route_id = f"{asset}:{buy_ex}:{buy_mt}->{sell_ex}:{sell_mt}"

        def _book_snapshot(sym: str, exchange: str, market: str):
            snap = ws_mgr.engine.get_snapshot(sym, exchange, market)
            if snap is None or not snap.is_valid():
                return None
            return {
                "bids": [[float(p), float(q)] for p, q in snap.bids[:depth]],
                "asks": [[float(p), float(q)] for p, q in snap.asks[:depth]],
                "ts": snap.timestamp,
            }

        # Engine stores market_type as "spot"/"futures" (with 's'),
        # but _norm_market_token returns "future" (without 's').
        # Map to engine-internal format for snapshot lookup.
        buy_mt_engine = "futures" if buy_mt == "future" else buy_mt
        sell_mt_engine = "futures" if sell_mt == "future" else sell_mt

        buy_book = _book_snapshot(asset, buy_ex, buy_mt_engine)
        sell_book = _book_snapshot(asset, sell_ex, sell_mt_engine)

        out_routes.append(
            {
                "route_id": route_id,
                "asset": asset,
                "buy_exchange": buy_ex,
                "buy_market": buy_mt,
                "sell_exchange": sell_ex,
                "sell_market": sell_mt,
                "buy_book": buy_book,
                "sell_book": sell_book,
                "ts_ms": now_ms,
            }
        )

    return web.json_response(
        {
            "schema_name": "scanner.export.orderbook",
            "schema_version": "1.0.0",
            "ts_ms": now_ms,
            "routes": out_routes,
        }
    )


# ---------------------------------------------------------------------------
# REST Handlers: Extra endpoints the frontend calls
# ---------------------------------------------------------------------------

async def handle_panels(request):
    if request.method == "GET":
        return web.json_response([])
    return web.json_response({"ok": True})


async def handle_buy_sell_data(request):
    ws_mgr = request.app.get("ws_manager")
    if not ws_mgr:
        return web.json_response({"entry": [], "exit": []})

    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    # Normalize to internal adapter slugs; UI sometimes sends display names.
    buy_ex = _norm_exchange(data.get("buy_exchange") or "")
    sell_ex = _norm_exchange(data.get("sell_exchange") or "")
    buy_symbol = (data.get("buy_symbol") or "").strip()
    sell_symbol = (data.get("sell_symbol") or "").strip()

    if not buy_ex or not sell_ex or not buy_symbol or not sell_symbol:
        return web.json_response(
            {"error": "Missing required fields"},
            status=400,
        )

    def _mt_from_symbol(sym: str) -> str:
        return "futures" if ":" in (sym or "") else "spot"

    buy_mt = _mt_from_symbol(buy_symbol)
    sell_mt = _mt_from_symbol(sell_symbol)

    # Hard rule: Spot must never be on the sell/short side.
    if sell_mt == "spot":
        return web.json_response(
            {"error": "Invalid pair: sell side cannot be SPOT"},
            status=400,
        )

    def _base_from_symbol(sym: str) -> str:
        s = (sym or "").strip()
        if not s:
            return ""
        s = s.split(":")[0]
        base = s.split("/")[0]
        return base.upper()

    base = _base_from_symbol(buy_symbol) or _base_from_symbol(sell_symbol)
    if not base:
        return web.json_response({"error": "Invalid symbols"}, status=400)

    def _parse_iso(ts: str) -> Optional[float]:
        if not ts:
            return None
        try:
            tss = ts.strip().replace("Z", "+00:00")
            return datetime.fromisoformat(tss).timestamp()
        except Exception:
            return None

    start_ts = _parse_iso(data.get("date_start") or "")
    end_ts = _parse_iso(data.get("date_end") or "")

    try:
        min_entry = float(data.get("min_entry_spread") or 0.0)
    except Exception:
        min_entry = 0.0
    try:
        min_exit = float(data.get("min_exit_spread") or 0.0)
    except Exception:
        min_exit = 0.0

    # Pull tracker history (decimated). We return both entry/exit series so the UI can render charts.
    history = ws_mgr.tracker.get_history(base, buy_ex, buy_mt, sell_ex, sell_mt, limit=5000)

    def _iso(ts_sec: float) -> str:
        return datetime.fromtimestamp(ts_sec, tz=timezone.utc).isoformat()

    entry_out = []
    exit_out = []
    for r in history:
        ts = float(r.get("timestamp") or 0.0)
        if start_ts is not None and ts < start_ts:
            continue
        if end_ts is not None and ts > end_ts:
            continue

        entry_spread = float(r.get("entry_spread") or 0.0)
        exit_spread = float(r.get("exit_spread") or 0.0)

        obj = {
            "time": _iso(ts),
            "entry_spread": round(entry_spread, 4),
            "exit_spread": round(exit_spread, 4),
        }

        # Thresholds are treated as minimum magnitude so negative values can still be displayed.
        if abs(entry_spread) >= max(min_entry, 0.0):
            entry_out.append(obj)
        if abs(exit_spread) >= max(min_exit, 0.0):
            exit_out.append(obj)

    return web.json_response({
        "entry": entry_out,
        "exit": exit_out,
        "meta": {
            "symbol": base,
            "buy_exchange": buy_ex,
            "buy_market": buy_mt,
            "sell_exchange": sell_ex,
            "sell_market": sell_mt,
            "records_total": len(history),
        },
    })


async def handle_telegram_send(request):
    return web.json_response({"ok": True, "message": "Telegram integration not configured"})


async def handle_discord_callback(request):
    return web.json_response({"ok": True})


async def handle_ui_snapshot(request):
    snapshot_path = OUT_DIR / "latest" / "ui_snapshot.json"
    if not snapshot_path.exists():
        snapshot_path = OUT_DIR / "ui_snapshot.json"
    if snapshot_path.exists():
        try:
            data = json.loads(snapshot_path.read_text(encoding="utf-8"))
            return web.json_response(data)
        except Exception:
            pass
    return web.json_response({"data": [], "error": "no_data"})


# ---------------------------------------------------------------------------
# WebSocket Handlers
# ---------------------------------------------------------------------------

async def handle_scanner_ws(request):
    # If not a WebSocket upgrade request, serve SPA HTML instead
    if request.headers.get("Upgrade", "").lower() != "websocket":
        return await handle_spa(request)
    ws = web.WebSocketResponse(heartbeat=45.0)
    await ws.prepare(request)

    ws_mgr = request.app.get("ws_manager")
    if ws_mgr:
        ws_mgr.add_scanner_client(ws)
    else:
        request.app.setdefault("scanner_websockets", set()).add(ws)

    # Send connection status (legacy-bundle compatible)
    await ws.send_json({"mensagem": "Conectado ao servidor de spreads", "tipo": "status"})

    logger.info("Scanner WebSocket connected")

    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                if msg.data == "ping":
                    await ws.send_str("pong")
            elif msg.type == WSMsgType.ERROR:
                logger.error(f"Scanner WS error: {ws.exception()}")
    finally:
        if ws_mgr:
            ws_mgr.remove_scanner_client(ws)
        else:
            request.app.get("scanner_websockets", set()).discard(ws)
        logger.info("Scanner WebSocket disconnected")

    return ws


async def handle_scanner_lite_ws(request):
    if request.headers.get("Upgrade", "").lower() != "websocket":
        return await handle_spa(request)
    ws = web.WebSocketResponse(heartbeat=45.0)
    await ws.prepare(request)

    ws_mgr = request.app.get("ws_manager")
    if ws_mgr:
        ws_mgr.add_scanner_lite_client(ws)

    await ws.send_json({"mensagem": "Conectado ao servidor de spreads (lite)", "tipo": "status"})
    if ws_mgr:
        snapshot = ws_mgr.get_current_scanner_lite_snapshot()
        if snapshot:
            await ws.send_json(snapshot)

    logger.info("Scanner lite WebSocket connected")

    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                if msg.data == "ping":
                    await ws.send_str("pong")
            elif msg.type == WSMsgType.ERROR:
                logger.error(f"Scanner lite WS error: {ws.exception()}")
    finally:
        if ws_mgr:
            ws_mgr.remove_scanner_lite_client(ws)
        logger.info("Scanner lite WebSocket disconnected")

    return ws


def _norm_exchange(ex: str) -> str:
    e = (ex or "").strip().lower()
    if not e:
        return ""
    # Handle common variations coming from the bundled frontend mappings.
    e_compact = (
        e.replace(" ", "")
        .replace(".", "")
        .replace("-", "")
        .replace("_", "")
    )
    if "gate" in e_compact:
        return "gate"
    if "kucoin" in e_compact:
        return "kucoin"
    if "bingx" in e_compact:
        return "bingx"
    if "mexc" in e_compact:
        return "mexc"
    if e_compact in ("xt", "xtcom"):
        return "xt"
    if "bitget" in e_compact:
        return "bitget"
    return e_compact


def _norm_market(market: str) -> str:
    m = (market or "").strip().lower()
    if m in ("future", "futures", "swap", "perp", "perpetual"):
        return "futures"
    return "spot"


async def handle_compare_ws(request):
    """WebSocket used by the legacy bundle for the PiP/details orderbook view."""
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    logger.info("Compare WebSocket connected")

    ws_mgr = request.app.get("ws_manager")
    if not ws_mgr:
        await ws.send_json({"type": "ERROR", "message": "Spread engine not running"})
        await ws.close()
        return ws

    sub = {
        "symbol": None,
        "buy_ex": None,
        "buy_mt": None,
        "sell_ex": None,
        "sell_mt": None,
    }
    stop = asyncio.Event()

    async def _sender():
        while not stop.is_set():
            await asyncio.sleep(0.5)
            if not sub["symbol"]:
                continue
            try:
                buy_snap = ws_mgr.engine.get_snapshot(
                    sub["symbol"], sub["buy_ex"], sub["buy_mt"]
                )
                sell_snap = ws_mgr.engine.get_snapshot(
                    sub["symbol"], sub["sell_ex"], sub["sell_mt"]
                )
                payload = {
                    "type": "DATA",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "buyExchange": {
                        "bids": buy_snap.bids if buy_snap else [],
                        "asks": buy_snap.asks if buy_snap else [],
                    },
                    "sellExchange": {
                        "bids": sell_snap.bids if sell_snap else [],
                        "asks": sell_snap.asks if sell_snap else [],
                    },
                }
                await ws.send_json(payload)
            except Exception:
                return

    sender_task = asyncio.create_task(_sender())

    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                if msg.data == "ping":
                    await ws.send_json({"type": "ping"})
                    continue
                try:
                    payload = json.loads(msg.data or "{}")
                except Exception:
                    continue
                if not isinstance(payload, dict):
                    continue

                symbol = (payload.get("crypto") or "").strip().upper()
                buy = payload.get("buy") or {}
                sell = payload.get("sell") or {}
                buy_ex = _norm_exchange(buy.get("exchange") or "")
                sell_ex = _norm_exchange(sell.get("exchange") or "")
                buy_mt = _norm_market(buy.get("market") or "")
                sell_mt = _norm_market(sell.get("market") or "")

                if not symbol or not buy_ex or not sell_ex:
                    await ws.send_json({"type": "ERROR", "message": "Invalid subscription payload"})
                    continue

                # Hard rule: Spot must never be the sell/short venue.
                if sell_mt == "spot":
                    await ws.send_json({"type": "ERROR", "message": "Invalid pair: sell side cannot be SPOT"})
                    continue

                sub.update(
                    {
                        "symbol": symbol,
                        "buy_ex": buy_ex,
                        "buy_mt": buy_mt,
                        "sell_ex": sell_ex,
                        "sell_mt": sell_mt,
                    }
                )
                await ws.send_json(
                    {
                        "type": "SUBSCRIBED",
                        "symbol": symbol,
                        "buy": {"exchange": buy_ex, "market": buy_mt},
                        "sell": {"exchange": sell_ex, "market": sell_mt},
                    }
                )

            elif msg.type == WSMsgType.ERROR:
                logger.error(f"Compare WS error: {ws.exception()}")
    finally:
        stop.set()
        sender_task.cancel()
        logger.info("Compare WebSocket disconnected")

    return ws


async def handle_monitor_ws(request):
    if request.headers.get("Upgrade", "").lower() != "websocket":
        return await handle_spa(request)
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    logger.info("Monitor WebSocket connected")

    ws_mgr = request.app.get("ws_manager")
    status = ws_mgr.get_status() if ws_mgr else {"status": "online"}

    await ws.send_json({
        "type": "STATUS",
        "payload": {
            "status": "online",
            "latency": 15,
            **status,
        },
    })

    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                if msg.data == "ping":
                    await ws.send_str("pong")
    finally:
        logger.info("Monitor WebSocket disconnected")
    return ws


async def handle_notifications_ws(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    logger.info("Notifications WebSocket connected")
    try:
        async for msg in ws:
            pass
    finally:
        logger.info("Notifications WebSocket disconnected")
    return ws


# ---------------------------------------------------------------------------
# SPA Fallback
# ---------------------------------------------------------------------------

async def handle_spa(request):
    path = request.match_info.get("tail", "")
    file_path = FRONTEND_DIR / path

    if path and file_path.is_file():
        return web.FileResponse(file_path)

    index_path = FRONTEND_DIR / "src-preview.html"
    if index_path.is_file():
        html = index_path.read_text(encoding="utf-8")
        return web.Response(
            text=html,
            content_type="text/html",
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )

    return web.Response(text="Frontend not found. Check 'novo frontend/frontend 2/' directory.", status=404)


async def _event_loop_lag_probe(app: web.Application):
    perf_monitor: RuntimePerfMonitor | None = app.get("perf_monitor")
    interval_sec = 0.1
    loop = asyncio.get_running_loop()
    expected = loop.time() + interval_sec
    while True:
        await asyncio.sleep(interval_sec)
        now = loop.time()
        lag_ms = max((now - expected) * 1000.0, 0.0)
        if perf_monitor is not None:
            perf_monitor.record_event_loop_lag(lag_ms)
        expected = now + interval_sec


async def _perf_snapshot_probe(app: web.Application):
    perf_monitor: RuntimePerfMonitor | None = app.get("perf_monitor")
    while True:
        await asyncio.sleep(30)
        ws_mgr = app.get("ws_manager")
        if perf_monitor is not None and ws_mgr and hasattr(ws_mgr, "get_perf_state"):
            perf_monitor.record_cache_state(ws_mgr.get_perf_state())


# ---------------------------------------------------------------------------
# Startup / Shutdown
# ---------------------------------------------------------------------------

async def on_startup(app):
    logger.info("Starting spread engine...")
    try:
        from spread.ws_manager import WSManager
        from spread.models import SpreadConfig
        from spread.runtime_audit import RuntimeAuditCollector

        config = SpreadConfig()
        # Optional runtime tuning via env (teamop.env / gemini.env).
        try:
            if os.environ.get("TEAM_OP_MAX_SYMBOLS"):
                config.max_symbols = int(os.environ["TEAM_OP_MAX_SYMBOLS"])
            if os.environ.get("TEAM_OP_DISCOVERY_MAX_SYMBOLS"):
                config.symbol_discovery_max_symbols = int(os.environ["TEAM_OP_DISCOVERY_MAX_SYMBOLS"])
            if os.environ.get("TEAM_OP_SYMBOL_DISCOVERY_ENABLED"):
                v = os.environ["TEAM_OP_SYMBOL_DISCOVERY_ENABLED"].strip().lower()
                config.symbol_discovery_enabled = v not in ("0", "false", "no", "off")
            if os.environ.get("TEAM_OP_TRACKER_RECORD_INTERVAL_SEC"):
                config.tracker_record_interval_sec = float(os.environ["TEAM_OP_TRACKER_RECORD_INTERVAL_SEC"])
            if os.environ.get("TEAM_OP_TRACKER_EPSILON_PCT"):
                config.tracker_epsilon_pct = float(os.environ["TEAM_OP_TRACKER_EPSILON_PCT"])
            if os.environ.get("TEAM_OP_TRACKER_MAX_RECORDS_PER_PAIR"):
                config.tracker_max_records_per_pair = int(os.environ["TEAM_OP_TRACKER_MAX_RECORDS_PER_PAIR"])
            if os.environ.get("TEAM_OP_TRACKER_GAP_THRESHOLD_SEC"):
                config.tracker_gap_threshold_sec = float(os.environ["TEAM_OP_TRACKER_GAP_THRESHOLD_SEC"])
            if os.environ.get("TEAM_OP_MIN_TOTAL_SPREAD_PCT"):
                config.min_total_spread_pct = float(os.environ["TEAM_OP_MIN_TOTAL_SPREAD_PCT"])
            if os.environ.get("TEAM_OP_TRACKER_DB_PATH"):
                config.tracker_db_path = os.environ["TEAM_OP_TRACKER_DB_PATH"].strip()
        except Exception as e:
            logger.warning(f"Invalid TEAM_OP_* tuning env vars: {e}")

        ws_mgr = WSManager(config)
        perf_monitor: RuntimePerfMonitor = app["perf_monitor"]
        ws_mgr.set_perf_monitor(perf_monitor)
        runtime_audit_dir = os.environ.get("TEAM_OP_RUNTIME_AUDIT_DIR", "").strip()
        if runtime_audit_dir:
            collector = RuntimeAuditCollector(
                output_dir=Path(runtime_audit_dir),
                record_interval_sec=float(config.tracker_record_interval_sec),
                gap_threshold_sec=float(getattr(config, "tracker_gap_threshold_sec", 0.0) or 0.0),
                duration_sec=int(float(os.environ.get("TEAM_OP_RUNTIME_AUDIT_DURATION_SEC", "7200"))),
            )
            ws_mgr.attach_runtime_audit(collector)
            app["runtime_audit"] = collector
        else:
            app["runtime_audit"] = None
        app["ws_manager"] = ws_mgr
        await ws_mgr.start()
        app["perf_tasks"] = [
            asyncio.create_task(_event_loop_lag_probe(app), name="event_loop_lag_probe"),
            asyncio.create_task(_perf_snapshot_probe(app), name="perf_snapshot_probe"),
        ]
        logger.info(f"Spread engine started: {len(config.symbols)} symbols, "
                     f"{len(config.exchanges)} exchanges")
    except Exception as e:
        logger.error(f"Failed to start spread engine: {e}")
        app["ws_manager"] = None
        app["runtime_audit"] = None


async def on_shutdown(app):
    perf_tasks = list(app.get("perf_tasks", []))
    for task in perf_tasks:
        task.cancel()
    if perf_tasks:
        await asyncio.gather(*perf_tasks, return_exceptions=True)

    ws_mgr = app.get("ws_manager")
    if ws_mgr:
        await ws_mgr.stop()
    for task in list(app.get("ml_training_tasks", {}).values()):
        task.cancel()

    logger.info("Server shut down")


# ---------------------------------------------------------------------------
# App Factory
# ---------------------------------------------------------------------------

_GZIP_TYPES = frozenset([
    "text/html", "text/css", "text/javascript", "text/plain",
    "application/javascript", "application/json", "application/xml",
    "image/svg+xml",
])
_GZIP_MIN_SIZE = 512  # don't bother compressing tiny responses
_gzip_cache: dict[str, bytes] = {}  # path -> compressed bytes (for static assets)


@web.middleware
async def gzip_middleware(request, handler):
    """Gzip-compress eligible responses when the client accepts it."""
    resp = await handler(request)
    accept = request.headers.get("Accept-Encoding", "")
    if "gzip" not in accept:
        return resp
    ct = (resp.content_type or "").split(";")[0].strip()
    if ct not in _GZIP_TYPES:
        return resp
    # Only compress regular responses (not StreamResponse / WS)
    if not isinstance(resp, web.Response) or resp.body is None:
        return resp
    body = resp.body
    if isinstance(body, bytes) and len(body) < _GZIP_MIN_SIZE:
        return resp
    raw = body if isinstance(body, bytes) else (body.encode("utf-8") if isinstance(body, str) else bytes(body))
    # Use cached compressed version for static assets
    cache_key = request.path if request.path.startswith("/assets/") else None
    if cache_key and cache_key in _gzip_cache:
        compressed = _gzip_cache[cache_key]
    else:
        compressed = await asyncio.to_thread(_gzip.compress, raw, 6)
        if len(compressed) >= len(raw):
            return resp  # compression didn't help
        if cache_key:
            _gzip_cache[cache_key] = compressed
    resp.body = compressed
    resp.headers["Content-Encoding"] = "gzip"
    resp.headers["Content-Length"] = str(len(compressed))
    resp.headers.pop("ETag", None)  # invalidate ETag since body changed
    return resp


@web.middleware
async def cache_middleware(request, handler):
    """Cache hashed assets long-term; never cache HTML (SPA entry point)."""
    resp = await handler(request)
    if request.path.startswith("/assets/"):
        # Hashed filenames (e.g. index-Bx06KCE-.js) are immutable — cache 7 days
        resp.headers["Cache-Control"] = "public, max-age=604800, immutable"
    ct = resp.content_type or ""
    if ct.startswith("text/html"):
        # HTML must always be re-validated to pick up new asset hashes
        resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp


def create_app() -> web.Application:
    app = web.Application(middlewares=[cors_middleware, gzip_middleware, cache_middleware])
    app["scanner_websockets"] = set()
    app["ml_training_tasks"] = {}
    app["perf_monitor"] = RuntimePerfMonitor()
    app["perf_tasks"] = []

    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)

    # ---- Auth routes ----
    app.router.add_post("/auth/login", handle_login)
    app.router.add_post("/auth/register", handle_register)
    app.router.add_post("/auth/otp/request", handle_otp_request)
    app.router.add_post("/auth/otp/verify", handle_otp_verify)
    app.router.add_get("/auth/profile", handle_profile)
    app.router.add_get("/auth/user", handle_profile)

    # Auth admin CRUD (stub)
    app.router.add_get("/auth/roles", handle_roles_list)
    app.router.add_post("/auth/roles", handle_generic_ok)
    app.router.add_get("/auth/permissions", handle_permissions_list)
    app.router.add_post("/auth/permissions", handle_generic_ok)
    app.router.add_get("/auth/users", handle_users_list)
    app.router.add_post("/auth/users", handle_generic_ok)
    app.router.add_post("/auth/users/filter", handle_users_list)

    # ---- Catalog routes ----
    app.router.add_get("/catalog/coins", handle_catalog_coins)
    app.router.add_get("/catalog/coins/names", handle_catalog_coins_names)
    app.router.add_post("/catalog/coins", handle_coins_crud)
    app.router.add_get("/catalog/coins/{id}/raw-data", handle_coin_raw_data)
    app.router.add_get("/catalog/exchanges", handle_catalog_exchanges)

    # ---- Finance routes ----
    app.router.add_get("/finance/plans", handle_finance_plans)
    app.router.add_get("/finance/payments", handle_finance_payments)
    app.router.add_get("/finance/licences", handle_finance_licences)

    # ---- User preferences ----
    app.router.add_get("/user/preferences", handle_user_preferences)
    app.router.add_put("/user/preferences", handle_user_preferences)

    # ---- Spread API ----
    app.router.add_get("/api/spread/opportunities", handle_spread_opportunities)
    app.router.add_get("/api/spread/opportunities-lite", handle_spread_opportunities_lite)
    app.router.add_get("/api/spread/opportunities-lite/summary", handle_spread_opportunities_lite_summary)
    app.router.add_get("/api/spread/opportunities/{pair_key}", handle_spread_opportunity_detail)
    app.router.add_get("/api/spread/status", handle_spread_status)
    app.router.add_get("/api/spread/debug/books", handle_debug_books)
    app.router.add_get("/api/spread/debug/status", handle_debug_status)
    app.router.add_get("/api/spread/debug/logs", handle_debug_logs)
    app.router.add_get("/api/debug/perf", handle_debug_perf)
    app.router.add_get("/api/spread/history/{symbol}", handle_spread_history)

    # ---- Export API (for ArbML / ML consumers) ----
    app.router.add_get("/_export/v1/opportunities", handle_export_opportunities)
    app.router.add_post("/_export/v1/history", handle_export_history)
    app.router.add_get("/_export/v1/funding_snapshot", handle_export_funding_snapshot)
    app.router.add_get("/_export/v1/filters_snapshot", handle_export_filters_snapshot)
    app.router.add_get("/_export/v1/networks", handle_export_networks)
    app.router.add_get("/_export/v1/funding_history", handle_export_funding_history)
    app.router.add_get("/_export/v1/futures_meta", handle_export_futures_meta)
    app.router.add_post("/_export/v1/orderbook", handle_export_orderbook)

    # ---- Funding (legacy) ----
    app.router.add_get("/api/funding/snapshot", handle_funding_snapshot)
    app.router.add_get("/ui", handle_ui_snapshot)

    # ---- Extra endpoints the frontend calls ----
    app.router.add_get("/panels", handle_panels)
    app.router.add_post("/panels", handle_panels)
    app.router.add_post("/api/v1/buy-sell-data", handle_buy_sell_data)
    app.router.add_post("/integrations/telegram/send", handle_telegram_send)
    app.router.add_get("/auth/discord/callback", handle_discord_callback)

    # ---- WebSockets ----
    app.router.add_get("/ws/scanner", handle_scanner_ws)
    app.router.add_get("/ws/scanner-lite", handle_scanner_lite_ws)
    app.router.add_get("/ws/monitor", handle_monitor_ws)
    app.router.add_get("/ws", handle_scanner_ws)
    app.router.add_get("/crypto/compare-ws", handle_compare_ws)
    app.router.add_get("/notifications/ws", handle_notifications_ws)
    app.router.add_get("/scanner", handle_scanner_ws)
    app.router.add_get("/monitor", handle_monitor_ws)

    # ---- Debug dashboard ----
    app.router.add_get("/debug", handle_debug_page)
    
    # ---- ML Advanced Dashboard ----
    app.router.add_get("/dashboard", handle_ml_dashboard_page)
    app.router.add_get("/dashboard/training", handle_ml_training_page)
    app.router.add_get("/api/v1/ml/dashboard", handle_ml_dashboard_api)
    app.router.add_get("/api/v1/ml/dashboard/summary", handle_ml_dashboard_summary_api)
    app.router.add_get("/api/v1/ml/dashboard/list", handle_ml_dashboard_list_api)
    app.router.add_get("/api/v1/ml/dashboard/{pair_key}", handle_ml_dashboard_detail_api)
    app.router.add_get("/api/v1/ml/training/sessions", handle_ml_training_sessions_api)
    app.router.add_patch("/api/v1/ml/training/sessions/{session_id}", handle_ml_training_session_patch)
    app.router.add_get("/api/v1/ml/training/sessions/{session_id}/exceptions", handle_ml_training_session_exceptions)
    app.router.add_post("/api/v1/ml/training/cohorts/preview", handle_ml_training_cohort_preview)
    app.router.add_get("/api/v1/ml/training/blocks", handle_ml_training_blocks_api)
    app.router.add_patch("/api/v1/ml/training/blocks/{block_id}", handle_ml_training_block_patch)
    app.router.add_post("/api/v1/ml/training/blocks/{block_id}/split", handle_ml_training_block_split)
    app.router.add_post("/api/v1/ml/training/blocks/{block_id}/merge-next", handle_ml_training_block_merge_next)
    app.router.add_patch("/api/v1/ml/training/config", handle_ml_training_config_patch)
    app.router.add_post("/api/v1/ml/training/resegment", handle_ml_training_resegment)
    app.router.add_post("/api/v1/ml/training/runs", handle_ml_training_run_create)
    app.router.add_get("/api/v1/ml/training/runs/{run_id}", handle_ml_training_run_status)
    app.router.add_get("/api/v1/ml/training/runs/latest", handle_ml_training_run_latest)

    # ---- Runtime frontend config ----
    async def handle_frontend_config(request):
        host = request.host
        scheme = "https" if request.secure else "http"
        ws_scheme = "wss" if request.secure else "ws"
        config_js = f"""window.__TEAMOP_CONFIG__ = {{
  API_BASE: "{scheme}://{host}",
  WS_BASE: "{ws_scheme}://{host}",
  WS_SCANNER: "{ws_scheme}://{host}/ws/scanner",
  WS_SCANNER_LITE: "{ws_scheme}://{host}/ws/scanner-lite",
  API_SCANNER_LITE: "{scheme}://{host}/api/spread/opportunities-lite",
  API_SCANNER_LITE_SUMMARY: "{scheme}://{host}/api/spread/opportunities-lite/summary",
  API_ML_DASHBOARD_SUMMARY: "{scheme}://{host}/api/v1/ml/dashboard/summary",
  API_ML_DASHBOARD_LIST: "{scheme}://{host}/api/v1/ml/dashboard/list",
  WATCHDOG_TIMEOUT_MS: 45000,
}};"""
        return web.Response(
            text=config_js,
            content_type="application/javascript",
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )

    app.router.add_get("/config.js", handle_frontend_config)

    # ---- Static assets (custom handler so gzip middleware can compress) ----
    if ASSETS_DIR.is_dir():
        _MIME_MAP = {
            ".js": "application/javascript",
            ".css": "text/css",
            ".json": "application/json",
            ".svg": "image/svg+xml",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".woff": "font/woff",
            ".woff2": "font/woff2",
            ".ttf": "font/ttf",
            ".ico": "image/x-icon",
            ".webp": "image/webp",
        }

        async def _serve_asset(request):
            fname = request.match_info["filename"]
            fpath = ASSETS_DIR / fname
            if not fpath.is_file() or ".." in fname:
                raise web.HTTPNotFound()
            ext = fpath.suffix.lower()
            ct = _MIME_MAP.get(ext, "application/octet-stream")
            body = fpath.read_bytes()
            return web.Response(body=body, content_type=ct)

        app.router.add_get("/assets/{filename:.+}", _serve_asset)

    # ---- SPA catch-all ----
    app.router.add_get("/", handle_spa)
    app.router.add_get("/{tail:.*}", handle_spa)

    return app


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Team OP Scanner Server")
    parser.add_argument("--host", default=HOST, help=f"Bind address (default: {HOST})")
    parser.add_argument("--port", type=int, default=PORT, help=f"Port (default: {PORT})")
    args = parser.parse_args()

    app = create_app()

    print(f"\n  Team OP Scanner")
    print(f"  http://{args.host}:{args.port}")
    print(f"  Frontend: {FRONTEND_DIR}")
    print(f"  Admin: {ADMIN_USER}\n")

    web.run_app(app, host=args.host, port=args.port, print=None)
