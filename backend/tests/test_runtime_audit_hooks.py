from pathlib import Path

from src.spread.models import SpreadConfig
from src.spread.orderbook import OrderBook
from src.spread.runtime_audit import RuntimeAuditCollector
from src.spread.spread_tracker import SpreadTracker
from src.spread.ws_manager import WSManager


def _read_lines(path: Path) -> list[str]:
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []


def test_spread_tracker_emits_runtime_audit_record_events(tmp_path: Path):
    collector = RuntimeAuditCollector(
        output_dir=tmp_path / "audit",
        record_interval_sec=0.0,
        gap_threshold_sec=60.0,
        duration_sec=60,
    )
    tracker = SpreadTracker(
        window_sec=3600,
        record_interval_sec=0.0,
        max_records_per_pair=0,
        epsilon_pct=0.0,
        history_enable_entry_spread_pct=0.0,
        track_enable_entry_spread_pct=0.0,
        db_path=tmp_path / "tracker.sqlite",
        gap_threshold_sec=60.0,
    )
    tracker.audit_collector = collector

    tracker.record_spread("BTC", "mexc", "spot", "gate", "futures", 0.20, -0.10, now_ts=100.0)
    tracker.record_spread("BTC", "mexc", "spot", "gate", "futures", 0.22, -0.08, now_ts=180.0)
    collector.finalize()

    alerts = "\n".join(_read_lines(collector.alerts_path))
    events = "\n".join(_read_lines(collector.events_path))
    assert "tracker_record" in events
    assert "gap_detected" in alerts


def test_ws_manager_on_book_update_emits_ws_ingest_metrics(tmp_path: Path):
    collector = RuntimeAuditCollector(
        output_dir=tmp_path / "audit",
        record_interval_sec=15.0,
        gap_threshold_sec=60.0,
        duration_sec=60,
    )
    config = SpreadConfig(exchanges=[], symbols=[], tracker_db_path=str(tmp_path / "tracker.sqlite"))
    ws_mgr = WSManager(config)
    ws_mgr.runtime_audit = collector

    book = OrderBook("mexc", "BTC", "spot")
    assert book.apply_snapshot([["100", "1"]], [["101", "1"]])
    ws_mgr._on_book_update("BTC", "mexc", "spot", book)
    collector.finalize()

    events = "\n".join(_read_lines(collector.events_path))
    assert "ws_ingest" in events
