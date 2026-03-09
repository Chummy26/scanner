import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from src.server import (
    handle_ml_training_cohort_preview,
    handle_ml_training_run_create,
    handle_ml_training_run_status,
    handle_ml_training_session_exceptions,
    handle_ml_training_session_patch,
    handle_ml_training_sessions_api,
)
from src.spread.spread_tracker import SpreadTracker


class _FakeRequest:
    def __init__(self, app: dict, *, match_info=None, query=None, payload=None):
        self.app = app
        self.match_info = match_info or {}
        self.query = query or {}
        self._payload = payload

    async def json(self):
        return self._payload if self._payload is not None else {}


class _FakeWSManager:
    def __init__(self, tracker: SpreadTracker, db_path: Path):
        self.tracker = tracker
        self.config = SimpleNamespace(tracker_db_path=str(db_path), tracker_gap_threshold_sec=tracker.gap_threshold_sec)


def _append_closed_session(db_path: Path, *, session_index: int) -> None:
    tracker = SpreadTracker(
        window_sec=10 * 24 * 60 * 60,
        record_interval_sec=15.0,
        max_records_per_pair=0,
        epsilon_pct=0.0,
        history_enable_entry_spread_pct=0.0,
        track_enable_entry_spread_pct=0.0,
        db_path=db_path,
        gap_threshold_sec=60.0,
    )
    base_ts = 1_700_000_000 + (session_index * 100_000)
    pairs = [
        (f"BTC{session_index}", "mexc", "spot", "gate", "futures", 0.30),
        (f"ETH{session_index}", "mexc", "spot", "gate", "futures", 0.25),
    ]
    for pair_offset, (symbol, buy_ex, buy_mt, sell_ex, sell_mt, offset) in enumerate(pairs):
        start = base_ts + (pair_offset * 1_000)
        for index in range(12):
            tracker.record_spread(
                symbol,
                buy_ex,
                buy_mt,
                sell_ex,
                sell_mt,
                offset + (index * 0.01),
                -0.10 + (index * 0.005),
                now_ts=float(start + (index * 15)),
            )
    assert tracker.flush_to_storage(now_ts=float(base_ts + 3_000), force=True)
    tracker.close_active_session(ended_at=float(base_ts + 3_000))


def _build_multi_session_tracker(tmp_path: Path) -> tuple[SpreadTracker, Path]:
    db_path = tmp_path / "tracker_history.sqlite"
    for session_index in range(4):
        _append_closed_session(db_path, session_index=session_index)
    tracker = SpreadTracker(
        window_sec=10 * 24 * 60 * 60,
        record_interval_sec=15.0,
        max_records_per_pair=0,
        epsilon_pct=0.0,
        history_enable_entry_spread_pct=0.0,
        track_enable_entry_spread_pct=0.0,
        db_path=db_path,
        gap_threshold_sec=60.0,
    )
    tracker.close_active_session()
    return tracker, db_path


def test_training_sessions_api_summarizes_sessions_and_exposes_exceptions(tmp_path: Path):
    tracker, db_path = _build_multi_session_tracker(tmp_path)
    first_session = tracker.list_training_sessions(include_open=False)["sessions"][0]
    first_block_id = first_session["blocks_preview"][0]["id"]
    tracker.update_training_block(first_block_id, selected_for_training=False, disabled_reason="manual_unselect")
    app = {"ws_manager": _FakeWSManager(tracker, db_path)}

    sessions_response = asyncio.run(handle_ml_training_sessions_api(_FakeRequest(app, query={"include_open": "0"})))
    sessions_payload = json.loads(sessions_response.text)
    assert sessions_payload["summary"]["total_sessions"] == 4
    assert sessions_payload["summary"]["auto_approved_sessions"] == 3
    assert sessions_payload["summary"]["review_state_counts"]["needs_review"] == 1
    assert sessions_payload["summary"]["review_state_counts"]["auto_approved"] == 3
    assert sessions_payload["sessions"][0]["pair_count"] == 2
    assert sessions_payload["sessions"][0]["symbol_count"] == 2
    assert sessions_payload["sessions"][0]["exception_block_count"] >= 1
    assert sessions_payload["sessions"][0]["approved_for_training"] is True
    assert sessions_payload["sessions"][0]["review_state"] == "needs_review"
    assert sessions_payload["sessions"][0]["dominant_exception_reason"] == "block_excluded"
    assert sessions_payload["sessions"][0]["exception_reason_counts"]["block_excluded"] >= 1
    assert sessions_payload["sessions"][1]["review_state"] == "auto_approved"

    exceptions_response = asyncio.run(
        handle_ml_training_session_exceptions(_FakeRequest(app, match_info={"session_id": str(first_session["id"])}))
    )
    exceptions_payload = json.loads(exceptions_response.text)
    assert exceptions_payload["session_id"] == first_session["id"]
    assert any(block["selected_for_training"] is False for block in exceptions_payload["blocks"])

    preview_response = asyncio.run(
        handle_ml_training_cohort_preview(
            _FakeRequest(app, payload={"sequence_length": 4, "prediction_horizon_sec": 60})
        )
    )
    preview_payload = json.loads(preview_response.text)
    preview_ids = [session["id"] for session in preview_payload["sessions"]]
    assert first_session["id"] not in preview_ids


def test_training_sessions_api_summary_only_omits_session_payloads(tmp_path: Path):
    tracker, db_path = _build_multi_session_tracker(tmp_path)
    app = {"ws_manager": _FakeWSManager(tracker, db_path)}

    response = asyncio.run(handle_ml_training_sessions_api(_FakeRequest(app, query={"include_open": "0", "summary_only": "1"})))
    payload = json.loads(response.text)

    assert payload["summary"]["total_sessions"] == 4
    assert payload["sessions"] == []


def test_training_session_patch_and_cohort_preview_follow_approved_sessions(tmp_path: Path):
    tracker, db_path = _build_multi_session_tracker(tmp_path)
    listing = tracker.list_training_sessions(include_open=False)
    excluded_session_id = listing["sessions"][1]["id"]
    app = {"ws_manager": _FakeWSManager(tracker, db_path)}

    patch_response = asyncio.run(
        handle_ml_training_session_patch(
            _FakeRequest(
                app,
                match_info={"session_id": str(excluded_session_id)},
                payload={"approved_for_training": False, "excluded_reason": "bad continuity"},
            )
        )
    )
    patch_payload = json.loads(patch_response.text)
    assert patch_payload["approved_for_training"] is False
    assert patch_payload["excluded_reason"] == "bad continuity"

    preview_response = asyncio.run(
        handle_ml_training_cohort_preview(
            _FakeRequest(app, payload={"sequence_length": 4, "prediction_horizon_sec": 60})
        )
    )
    preview_payload = json.loads(preview_response.text)
    approved_session_ids = [session["id"] for session in preview_payload["sessions"]]
    assert excluded_session_id not in approved_session_ids
    assert preview_payload["mode"] == "expanding_walk_forward"
    assert preview_payload["folds"]
    assert preview_payload["folds"][-1]["train_session_ids"]
    assert preview_payload["folds"][-1]["validation_session_ids"]
    assert preview_payload["folds"][-1]["test_session_ids"]
    assert preview_payload["summary"]["eligible_sessions"] == len(approved_session_ids)
    assert preview_payload["summary"]["coverage_start_ts"] > 0
    assert preview_payload["summary"]["coverage_end_ts"] >= preview_payload["summary"]["coverage_start_ts"]


def test_training_run_create_uses_session_ids_and_snapshots_sessions(tmp_path: Path, monkeypatch):
    tracker, db_path = _build_multi_session_tracker(tmp_path)
    session_ids = [session["id"] for session in tracker.list_training_sessions(include_open=False)["sessions"][:3]]
    app = {"ws_manager": _FakeWSManager(tracker, db_path), "ml_training_tasks": {}}
    queued_coroutines = []

    async def _fake_execute_training_run(app_obj, run_id):
        tracker.update_training_run(run_id, status="completed", finished_at=123.0, result={"ok": True})

    monkeypatch.setattr("src.server._execute_training_run", _fake_execute_training_run)
    monkeypatch.setattr("src.server.asyncio.create_task", lambda coro: queued_coroutines.append(coro) or SimpleNamespace(done=lambda: True))

    create_response = asyncio.run(
        handle_ml_training_run_create(
            _FakeRequest(
                app,
                payload={"session_ids": session_ids, "sequence_length": 4, "prediction_horizon_sec": 60},
            )
        )
    )
    create_payload = json.loads(create_response.text)
    assert create_response.status == 202
    assert [session["session_id"] for session in create_payload["sessions"]] == session_ids
    assert create_payload["blocks"]

    asyncio.run(queued_coroutines[0])
    status_response = asyncio.run(handle_ml_training_run_status(_FakeRequest(app, match_info={"run_id": str(create_payload["id"])})))
    status_payload = json.loads(status_response.text)
    assert status_payload["status"] == "completed"
    assert [session["session_id"] for session in status_payload["sessions"]] == session_ids


def test_training_run_create_rejects_sequence_longer_than_available_blocks(tmp_path: Path):
    tracker, db_path = _build_multi_session_tracker(tmp_path)
    app = {"ws_manager": _FakeWSManager(tracker, db_path), "ml_training_tasks": {}}
    session_ids = [session["id"] for session in tracker.list_training_sessions(include_open=False)["sessions"][:2]]

    response = asyncio.run(
        handle_ml_training_run_create(
            _FakeRequest(
                app,
                payload={"session_ids": session_ids, "sequence_length": 15, "prediction_horizon_sec": 240},
            )
        )
    )
    payload = json.loads(response.text)

    assert response.status == 400
    assert "clean-cycle quality gates" in payload["error"] or "No selected blocks available" in payload["error"]
