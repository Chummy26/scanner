import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from src.server import (
    _execute_training_run,
    handle_ml_training_block_patch,
    handle_ml_training_block_split,
    handle_ml_training_blocks_api,
    handle_ml_training_run_create,
    handle_ml_training_run_latest,
    handle_ml_training_run_status,
)
from src.spread.spread_tracker import SpreadTracker


class _FakeRequest:
    def __init__(self, app: dict, *, match_info=None, query=None, payload=None):
        self.app = app
        self.match_info = match_info or {}
        self.query = query or {}
        self._payload = payload

    async def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload if self._payload is not None else {}


class _FakeWSManager:
    def __init__(self, tracker: SpreadTracker, db_path: Path):
        self.tracker = tracker
        self.config = SimpleNamespace(tracker_db_path=str(db_path), tracker_gap_threshold_sec=tracker.gap_threshold_sec)


def _build_tracker(tmp_path: Path) -> tuple[SpreadTracker, Path]:
    db_path = tmp_path / "tracker_history.sqlite"
    tracker = SpreadTracker(
        window_sec=7200,
        record_interval_sec=15.0,
        max_records_per_pair=0,
        epsilon_pct=0.0,
        history_enable_entry_spread_pct=0.0,
        track_enable_entry_spread_pct=0.0,
        db_path=db_path,
        gap_threshold_sec=45.0,
    )
    pair = ("BTC", "mexc", "spot", "gate", "futures")
    for ts in (0.0, 15.0, 30.0, 90.0, 105.0):
        tracker.record_spread(*pair, 0.40, -0.20, now_ts=ts)
    assert tracker.flush_to_storage(now_ts=105.0, force=True)
    tracker.close_active_session(ended_at=105.0)
    return tracker, db_path


def test_training_blocks_api_lists_real_sqlite_sessions_and_blocks(tmp_path: Path):
    tracker, db_path = _build_tracker(tmp_path)
    app = {"ws_manager": _FakeWSManager(tracker, db_path)}

    response = asyncio.run(handle_ml_training_blocks_api(_FakeRequest(app)))
    payload = json.loads(response.text)

    assert payload["summary"]["total_sessions"] == 1
    assert payload["summary"]["total_blocks"] == 2
    assert payload["summary"]["trainable_blocks"] == 1
    assert payload["sessions"][0]["status"] == "closed"
    assert [block["boundary_reason"] for block in payload["sessions"][0]["blocks"]] == ["initial", "auto_gap"]


def test_training_blocks_api_rejects_invalid_session_id(tmp_path: Path):
    tracker, db_path = _build_tracker(tmp_path)
    app = {"ws_manager": _FakeWSManager(tracker, db_path)}

    response = asyncio.run(handle_ml_training_blocks_api(_FakeRequest(app, query={"session_id": "invalid"})))

    assert response.status == 400
    assert "error" in json.loads(response.text)


def test_training_blocks_api_summary_only_omits_block_payloads(tmp_path: Path):
    tracker, db_path = _build_tracker(tmp_path)
    app = {"ws_manager": _FakeWSManager(tracker, db_path)}

    response = asyncio.run(handle_ml_training_blocks_api(_FakeRequest(app, query={"summary_only": "1"})))
    payload = json.loads(response.text)

    assert payload["summary"]["total_sessions"] == 1
    assert payload["summary"]["total_blocks"] == 2
    assert payload["sessions"][0]["blocks"] == []


def test_training_block_handlers_patch_and_reject_invalid_split(tmp_path: Path):
    tracker, db_path = _build_tracker(tmp_path)
    app = {"ws_manager": _FakeWSManager(tracker, db_path)}
    block_id = tracker.list_training_blocks()["sessions"][0]["blocks"][0]["id"]

    patch_response = asyncio.run(
        handle_ml_training_block_patch(
            _FakeRequest(
                app,
                match_info={"block_id": str(block_id)},
                payload={"selected_for_training": False, "notes": "manual review"},
            )
        )
    )
    patched_payload = json.loads(patch_response.text)
    assert patched_payload["selected_for_training"] is False
    assert patched_payload["notes"] == "manual review"

    split_response = asyncio.run(
        handle_ml_training_block_split(
            _FakeRequest(
                app,
                match_info={"block_id": str(block_id)},
                payload={"split_ts": -1.0},
            )
        )
    )
    assert split_response.status == 400
    assert "error" in json.loads(split_response.text)


def test_training_run_handlers_snapshot_selected_blocks(tmp_path: Path, monkeypatch):
    tracker, db_path = _build_tracker(tmp_path)
    app = {"ws_manager": _FakeWSManager(tracker, db_path), "ml_training_tasks": {}}
    session_id = tracker.list_training_sessions(include_open=False)["sessions"][0]["id"]
    tracker.update_training_session(session_id, approved_for_training=True, notes="reviewed for training")
    selected_block_ids = tracker.get_selected_training_block_ids(trainable_only=True)
    queued_coroutines = []

    async def _fake_execute_training_run(app_obj, run_id):
        tracker.update_training_run(run_id, status="completed", finished_at=123.0, result={"ok": True})

    monkeypatch.setattr("src.server._execute_training_run", _fake_execute_training_run)
    monkeypatch.setattr("src.server.asyncio.create_task", lambda coro: queued_coroutines.append(coro) or SimpleNamespace(done=lambda: True))

    create_response = asyncio.run(
        handle_ml_training_run_create(
                _FakeRequest(
                    app,
                    payload={"sequence_length": 2, "prediction_horizon_sec": 30},
                )
            )
        )
    create_payload = json.loads(create_response.text)
    assert create_response.status == 202
    assert [session["session_id"] for session in create_payload["sessions"]] == [session_id]
    assert [block["block_id"] for block in create_payload["blocks"]] == selected_block_ids

    assert len(queued_coroutines) == 1
    asyncio.run(queued_coroutines[0])

    status_response = asyncio.run(
        handle_ml_training_run_status(
            _FakeRequest(app, match_info={"run_id": str(create_payload["id"])})
        )
    )
    status_payload = json.loads(status_response.text)
    assert status_payload["status"] == "completed"
    assert status_payload["result"]["ok"] is True
    assert [block["block_id"] for block in status_payload["blocks"]] == selected_block_ids

    latest_response = asyncio.run(handle_ml_training_run_latest(_FakeRequest(app)))
    latest_payload = json.loads(latest_response.text)
    assert latest_response.status == 200
    assert latest_payload["id"] == create_payload["id"]
    assert latest_payload["status"] == "completed"


def test_execute_training_run_uses_clean_training_cycle(tmp_path: Path, monkeypatch):
    tracker, db_path = _build_tracker(tmp_path)
    app = {"ws_manager": _FakeWSManager(tracker, db_path), "ml_training_tasks": {}}
    session_id = tracker.list_training_sessions(include_open=False)["sessions"][0]["id"]
    tracker.update_training_session(session_id, approved_for_training=True, notes="reviewed for training")
    run_payload = tracker.create_training_run(
        session_ids=[session_id],
        sequence_length=2,
        prediction_horizon_sec=30,
    )
    called: dict[str, object] = {}

    def _fake_clean_training_cycle(**kwargs):
        called.update(kwargs)
        return {
            "model_status": "trained",
            "preflight": {"selected_threshold": 0.8},
            "training": {"selected_threshold": 0.8},
            "artifacts": {"audit_path": str(tmp_path / "audit.md")},
        }

    def _fail_run_training_loop(**_kwargs):
        raise AssertionError("run_training_loop should not be used by server training runs")

    monkeypatch.setattr("spread.train_model.run_clean_training_cycle", _fake_clean_training_cycle)
    monkeypatch.setattr("spread.train_model.run_training_loop", _fail_run_training_loop)

    asyncio.run(_execute_training_run(app, int(run_payload["id"])))
    updated = tracker.get_training_run(int(run_payload["id"]))

    assert updated["status"] == "completed"
    assert updated["result"]["preflight"]["selected_threshold"] == 0.8
    assert Path(called["artifact_dir"]).name == f"run_{int(run_payload['id'])}"
