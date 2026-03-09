import argparse
import json
import sqlite3
import urllib.request
from pathlib import Path


def _fetch_json(base_url: str, path: str) -> dict:
    with urllib.request.urlopen(f"{base_url}{path}", timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _post_json(base_url: str, path: str, payload: dict | None = None) -> dict:
    request = urllib.request.Request(
        f"{base_url}{path}",
        data=json.dumps(payload or {}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _fetch_text(base_url: str, path: str) -> str:
    with urllib.request.urlopen(f"{base_url}{path}", timeout=30) as response:
        return response.read().decode("utf-8")


def inspect_runtime(db_path: Path, base_url: str, run_id: int | None = None) -> dict:
    sessions_payload = _fetch_json(base_url, "/api/v1/ml/training/sessions?include_open=1&summary_only=1")
    preview_payload = _post_json(base_url, "/api/v1/ml/training/cohorts/preview", {})
    blocks_payload = _fetch_json(base_url, "/api/v1/ml/training/blocks?summary_only=1")
    quality_payload = _fetch_json(base_url, "/api/v1/ml/training/quality-report?summary_only=1")
    html = _fetch_text(base_url, "/dashboard/training")

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        sessions = list(
            conn.execute(
                """
                SELECT
                    s.id,
                    s.status,
                    s.started_at,
                    s.ended_at,
                    s.approved_for_training,
                    s.excluded_reason,
                    COALESCE(MIN(b.start_ts), 0.0) AS data_start_ts,
                    COALESCE(MAX(b.end_ts), 0.0) AS data_end_ts,
                    COALESCE(COUNT(b.id), 0) AS block_count,
                    COALESCE(SUM(CASE WHEN b.selected_for_training = 1 AND b.is_open = 0 AND b.record_count > 1 THEN 1 ELSE 0 END), 0) AS trainable_block_count
                FROM tracker_capture_sessions s
                LEFT JOIN tracker_pair_blocks b ON b.session_id = s.id
                GROUP BY s.id
                HAVING COUNT(b.id) > 0
                ORDER BY COALESCE(MAX(b.end_ts), s.ended_at, s.started_at) DESC, s.id DESC
                """
            )
        )
        blocks = list(
            conn.execute(
                """
                SELECT id, session_id, start_ts, end_ts, record_count, max_gap_sec, boundary_reason,
                       selected_for_training, is_open
                FROM tracker_pair_blocks
                ORDER BY session_id ASC, id ASC
                """
            )
        )
        run_rows = list(
            conn.execute(
                """
                SELECT id, status, selected_block_count, sequence_length, prediction_horizon_sec, artifact_dir
                FROM ml_training_runs
                ORDER BY id ASC
                """
            )
        )
        run_blocks = list(
            conn.execute(
                """
                SELECT run_id, block_id, session_id
                FROM ml_training_run_blocks
                ORDER BY run_id ASC, session_id ASC, block_id ASC
                """
            )
        )
        run_sessions = list(
            conn.execute(
                """
                SELECT run_id, session_id, position
                FROM ml_training_run_sessions
                ORDER BY run_id ASC, position ASC, session_id ASC
                """
            )
        )
        record_count_mismatches = []
        max_gap_mismatches = []
        zero_record_blocks = []
        range_mismatches = []
        missing_event_blocks = []
        open_blocks_after_close = []
        for block in blocks:
            timestamps = [
                float(row["ts"])
                for row in conn.execute("SELECT ts FROM tracker_records WHERE block_id = ? ORDER BY ts ASC", (int(block["id"]),))
            ]
            if int(block["record_count"]) <= 0:
                zero_record_blocks.append({"block_id": int(block["id"])})
            if len(timestamps) != int(block["record_count"]):
                record_count_mismatches.append(
                    {
                        "block_id": int(block["id"]),
                        "expected_record_count": int(block["record_count"]),
                        "observed_record_count": len(timestamps),
                    }
                )
            observed_gap = max((curr - prev for prev, curr in zip(timestamps, timestamps[1:])), default=0.0)
            if abs(observed_gap - float(block["max_gap_sec"])) > 1e-9:
                max_gap_mismatches.append(
                    {
                        "block_id": int(block["id"]),
                        "expected_max_gap_sec": float(block["max_gap_sec"]),
                        "observed_max_gap_sec": float(observed_gap),
                    }
                )
            if timestamps and (
                abs(float(min(timestamps)) - float(block["start_ts"])) > 1e-9
                or abs(float(max(timestamps)) - float(block["end_ts"])) > 1e-9
            ):
                range_mismatches.append(
                    {
                        "block_id": int(block["id"]),
                        "expected_start_ts": float(block["start_ts"]),
                        "observed_start_ts": float(min(timestamps)),
                        "expected_end_ts": float(block["end_ts"]),
                        "observed_end_ts": float(max(timestamps)),
                    }
                )
        open_blocks_after_close = [
            dict(row)
            for row in conn.execute(
                """
                SELECT b.id AS block_id, b.session_id
                FROM tracker_pair_blocks b
                JOIN tracker_capture_sessions s ON s.id = b.session_id
                WHERE b.is_open = 1 AND s.status != 'open'
                ORDER BY b.session_id ASC, b.id ASC
                """
            )
        ]
        missing_event_blocks = [
            dict(row)
            for row in conn.execute(
                """
                SELECT e.pair_id, e.event_type, e.ts, e.session_id, e.block_id
                FROM tracker_events e
                LEFT JOIN tracker_pair_blocks b ON b.id = e.block_id
                WHERE e.block_id IS NULL
                   OR b.id IS NULL
                   OR b.session_id != e.session_id
                   OR e.ts < b.start_ts
                   OR e.ts > b.end_ts
                ORDER BY e.session_id ASC, e.ts ASC
                """
            )
        ]

    selected_session_ids_api = [int(session["id"]) for session in preview_payload.get("sessions", [])]
    selected_sessions_db = [
        session
        for session in sorted(sessions, key=lambda row: (float(row["data_end_ts"]), int(row["id"])))
        if int(session["approved_for_training"]) == 1 and str(session["status"]) != "open" and int(session["trainable_block_count"]) > 0
    ]
    selected_session_ids_db = [int(session["id"]) for session in selected_sessions_db]
    selected_block_ids_api = [
        int(block["id"])
        for session in blocks_payload.get("sessions", [])
        if int(session.get("id") or 0) in set(selected_session_ids_api)
        for block in session.get("blocks", [])
        if block.get("selected_for_training") and block.get("trainable")
    ]
    selected_block_ids_db = [
        int(block["id"])
        for block in blocks
        if int(block["selected_for_training"]) == 1
        and int(block["is_open"]) == 0
        and int(block["record_count"]) > 1
        and int(block["session_id"]) in set(selected_session_ids_db)
    ]
    run_snapshot_session_ids = [
        int(row["session_id"])
        for row in run_sessions
        if run_id is None or int(row["run_id"]) == int(run_id)
    ]
    run_snapshot_ids = [
        int(row["block_id"])
        for row in run_blocks
        if run_id is None or int(row["run_id"]) == int(run_id)
    ]

    return {
        "html_has_training_title": "Sessões contínuas do LSTM" in html,
        "html_has_sessions_api": "/api/v1/ml/training/sessions" in html,
        "html_has_preview_api": "/api/v1/ml/training/cohorts/preview" in html,
        "html_has_blocks_api": "/api/v1/ml/training/blocks" in html,
        "sessions_api_summary": sessions_payload.get("summary", {}),
        "preview_summary": preview_payload.get("summary", {}),
        "blocks_api_summary": blocks_payload.get("summary", {}),
        "quality_summary": quality_payload.get("summary", {}),
        "sessions": [dict(row) for row in sessions],
        "blocks_total": len(blocks),
        "selected_session_ids_api": selected_session_ids_api,
        "selected_session_ids_db": selected_session_ids_db,
        "selected_block_ids_api": selected_block_ids_api,
        "selected_block_ids_db": selected_block_ids_db,
        "run_snapshot_session_ids": run_snapshot_session_ids,
        "run_snapshot_ids": run_snapshot_ids,
        "sessions_match_db": selected_session_ids_api == selected_session_ids_db,
        "selection_matches_db": sorted(selected_block_ids_api) == sorted(selected_block_ids_db),
        "snapshot_sessions_match": run_snapshot_session_ids == (selected_session_ids_db if run_id is not None else selected_session_ids_api),
        "snapshot_matches_selection": sorted(run_snapshot_ids) == sorted(selected_block_ids_db if run_id is not None else selected_block_ids_api),
        "record_count_mismatches": record_count_mismatches,
        "max_gap_mismatches": max_gap_mismatches,
        "zero_record_blocks": zero_record_blocks,
        "range_mismatches": range_mismatches,
        "missing_event_blocks": missing_event_blocks,
        "open_blocks_after_close": open_blocks_after_close,
        "runs": [dict(row) for row in run_rows],
    }


def build_report(result: dict) -> str:
    sessions = result.get("sessions", [])
    runs = result.get("runs", [])
    return (
        "# Training Sessions Runtime Audit\n\n"
        "## UI\n"
        f"- Training page title present: {result.get('html_has_training_title')}\n"
        f"- Training page references sessions API: {result.get('html_has_sessions_api')}\n"
        f"- Training page references cohort preview API: {result.get('html_has_preview_api')}\n"
        f"- Training page references blocks API: {result.get('html_has_blocks_api')}\n\n"
        "## Sessions API Summary\n"
        f"- Total sessions: {int(result.get('sessions_api_summary', {}).get('total_sessions', 0))}\n"
        f"- Approved sessions: {int(result.get('sessions_api_summary', {}).get('approved_sessions', 0))}\n"
        f"- Trainable sessions: {int(result.get('sessions_api_summary', {}).get('trainable_sessions', 0))}\n"
        f"- Exception sessions: {int(result.get('sessions_api_summary', {}).get('exception_sessions', 0))}\n\n"
        "## Cohort Preview Summary\n"
        f"- Eligible sessions: {int(result.get('preview_summary', {}).get('eligible_sessions', 0))}\n"
        f"- Eligible blocks: {int(result.get('preview_summary', {}).get('eligible_blocks', 0))}\n"
        f"- Fold count: {int(result.get('preview_summary', {}).get('fold_count', 0))}\n\n"
        "## Blocks API Summary\n"
        f"- Total sessions: {int(result.get('blocks_api_summary', {}).get('total_sessions', 0))}\n"
        f"- Total blocks: {int(result.get('blocks_api_summary', {}).get('total_blocks', 0))}\n"
        f"- Trainable blocks: {int(result.get('blocks_api_summary', {}).get('trainable_blocks', 0))}\n\n"
        "## Quality Report Summary\n"
        f"- Critical sessions: {int(result.get('quality_summary', {}).get('critical_sessions', 0))}\n"
        f"- Training-ready sessions: {int(result.get('quality_summary', {}).get('training_ready_sessions', 0))}\n"
        f"- Burn-in passed sessions: {int(result.get('quality_summary', {}).get('burn_in_passed_sessions', 0))}\n"
        f"- Zero-record blocks: {int(result.get('quality_summary', {}).get('zero_record_blocks', 0))}\n"
        f"- Record-count mismatches: {int(result.get('quality_summary', {}).get('record_count_mismatches', 0))}\n"
        f"- Range mismatches: {int(result.get('quality_summary', {}).get('range_mismatches', 0))}\n"
        f"- Missing event blocks: {int(result.get('quality_summary', {}).get('missing_event_blocks', 0))}\n\n"
        "## Sessions\n"
        + ("\n".join(
            f"- Session {int(session['id'])}: status={session['status']} approved={bool(session['approved_for_training'])} data={float(session['data_start_ts']):.3f}->{float(session['data_end_ts']):.3f} blocks={int(session['block_count'])} trainable={int(session['trainable_block_count'])}"
            for session in sessions
        ) if sessions else "- No sessions persisted")
        + "\n\n## Selection Consistency\n"
        f"- Selected session ids from preview: {result.get('selected_session_ids_api', [])}\n"
        f"- Selected session ids from DB: {result.get('selected_session_ids_db', [])}\n"
        f"- Training run snapshot session ids: {result.get('run_snapshot_session_ids', [])}\n"
        f"- Session selection matches DB: {result.get('sessions_match_db')}\n"
        f"- Selected block ids from API: {result.get('selected_block_ids_api', [])}\n"
        f"- Selected block ids from DB: {result.get('selected_block_ids_db', [])}\n"
        f"- Training run snapshot block ids: {result.get('run_snapshot_ids', [])}\n"
        f"- API selection matches DB: {result.get('selection_matches_db')}\n"
        f"- Training snapshot matches selected sessions: {result.get('snapshot_sessions_match')}\n"
        f"- Training snapshot matches selected blocks: {result.get('snapshot_matches_selection')}\n\n"
        "## SQLite Integrity\n"
        f"- Blocks total: {int(result.get('blocks_total', 0))}\n"
        f"- Zero-record blocks: {len(result.get('zero_record_blocks', []))}\n"
        f"- Record count mismatches: {len(result.get('record_count_mismatches', []))}\n"
        f"- Max gap mismatches: {len(result.get('max_gap_mismatches', []))}\n\n"
        f"- Range mismatches: {len(result.get('range_mismatches', []))}\n"
        f"- Missing event blocks: {len(result.get('missing_event_blocks', []))}\n"
        f"- Open blocks after close: {len(result.get('open_blocks_after_close', []))}\n\n"
        "## Runs\n"
        + ("\n".join(
            f"- Run {int(run['id'])}: status={run['status']} selected_block_count={int(run['selected_block_count'])} sequence_length={int(run['sequence_length'])} horizon={int(run['prediction_horizon_sec'])}"
            for run in runs
        ) if runs else "- No runs recorded")
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description="Verify SQLite training sessions, UI and run snapshots.")
    parser.add_argument("--db-path", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--run-id", type=int, default=None)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    result = inspect_runtime(Path(args.db_path), args.base_url, run_id=args.run_id)
    report = build_report(result)
    if args.output:
        Path(args.output).write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
