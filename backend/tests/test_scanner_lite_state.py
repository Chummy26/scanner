from src.spread.models import SpreadConfig, SpreadOpportunity
from src.spread.ws_manager import WSManager


def _make_opportunity(index: int) -> SpreadOpportunity:
    return SpreadOpportunity(
        asset=f"COIN{index}",
        arb_type="spot_futures",
        buy_exchange="mexc",
        sell_exchange="gate",
        buy_market_type="spot",
        sell_market_type="futures",
        buy_price=100.0 + index,
        sell_price=101.0 + index,
        entry_spread_pct=1.0 + (index * 0.001),
        exit_spread_pct=-0.4,
    )


def test_scanner_lite_snapshot_and_delta_remain_bounded_and_remove_missing_rows(tmp_path):
    cfg = SpreadConfig(exchanges=[], symbols=[], tracker_db_path=str(tmp_path / "tracker.sqlite"))
    ws_mgr = WSManager(cfg)

    batch_one = [_make_opportunity(index) for index in range(600)]
    ws_mgr._refresh_scanner_lite_state(batch_one)

    assert len(ws_mgr.get_current_opportunities_lite()) == 500
    snapshot = ws_mgr.get_current_scanner_lite_snapshot()
    assert snapshot["count"] == 500
    assert snapshot["total_count"] == 600

    kept = batch_one[:400]
    ws_mgr._refresh_scanner_lite_state(kept)
    delta = dict(ws_mgr._last_scanner_lite_delta)

    assert len(ws_mgr.get_current_opportunities_lite()) == 400
    assert delta["total_count"] == 400
    assert delta["remove_count"] == 100
    assert len(delta["removes"]) == 100
