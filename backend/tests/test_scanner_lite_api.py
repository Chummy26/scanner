import asyncio
import json

from src.server import (
    handle_scanner_lite_ws,
    handle_spread_opportunities_lite,
    handle_spread_opportunities_lite_summary,
    handle_spread_opportunity_detail,
)


class _FakeRequest:
    def __init__(self, app: dict, *, match_info=None, headers=None):
        self.app = app
        self.match_info = match_info or {}
        self.headers = headers or {}


class _FakeWSManager:
    def __init__(self, rows, summary):
        self._rows = rows
        self._summary = summary

    def get_current_opportunities_lite(self):
        return list(self._rows)

    def get_current_opportunities_lite_summary(self):
        return dict(self._summary)

    def get_scanner_opportunity_detail(self, pair_key):
        for row in self._rows:
            if row["pairKey"] == pair_key:
                return {"pairKey": pair_key, "symbol": row["symbol"], "details": True}
        return None

    def get_status(self):
        return {"running": True, "opportunities": len(self._rows)}

    def get_current_scanner_lite_snapshot(self):
        return {"type": "arbitrage_data_lite_snapshot", "data": list(self._rows), "summary": dict(self._summary)}

    def add_scanner_lite_client(self, _ws):
        return None

    def remove_scanner_lite_client(self, _ws):
        return None


def test_scanner_lite_opportunities_endpoint_returns_light_payload_and_summary():
    rows = [
        {
            "pairKey": "AAA|mexc|spot|gate|futures",
            "symbol": "AAA",
            "entrySpread": 1.5,
            "signalAction": "EXECUTE",
        }
    ]
    request = _FakeRequest({"ws_manager": _FakeWSManager(rows, {"total": 1, "signals": {"EXECUTE": 1}})})

    response = asyncio.run(handle_spread_opportunities_lite(request))
    payload = json.loads(response.text)

    assert payload["data"] == rows
    assert payload["summary"]["total"] == 1
    assert payload["status"]["running"] is True


def test_scanner_lite_summary_endpoint_returns_summary_only():
    request = _FakeRequest({"ws_manager": _FakeWSManager([], {"total": 0, "signals": {}})})

    response = asyncio.run(handle_spread_opportunities_lite_summary(request))
    payload = json.loads(response.text)

    assert payload == {"summary": {"total": 0, "signals": {}}}


def test_scanner_lite_detail_endpoint_returns_full_detail_by_pair_key():
    rows = [{"pairKey": "AAA|mexc|spot|gate|futures", "symbol": "AAA"}]
    request = _FakeRequest(
        {"ws_manager": _FakeWSManager(rows, {"total": 1})},
        match_info={"pair_key": "AAA|mexc|spot|gate|futures"},
    )

    response = asyncio.run(handle_spread_opportunity_detail(request))
    payload = json.loads(response.text)

    assert payload["data"]["pairKey"] == "AAA|mexc|spot|gate|futures"
    assert payload["data"]["details"] is True


def test_scanner_lite_detail_endpoint_returns_404_for_unknown_pair():
    request = _FakeRequest(
        {"ws_manager": _FakeWSManager([], {"total": 0})},
        match_info={"pair_key": "missing|pair"},
    )

    response = asyncio.run(handle_spread_opportunity_detail(request))

    assert response.status == 404


def test_scanner_lite_non_websocket_upgrade_falls_back_to_spa():
    request = _FakeRequest({"ws_manager": _FakeWSManager([], {"total": 0})}, headers={})

    response = asyncio.run(handle_scanner_lite_ws(request))

    assert response.status == 200
