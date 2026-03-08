from tests.verify_training_blocks_runtime import build_report


def test_build_report_includes_runtime_sections():
    report = build_report(
        {
            "html_has_training_title": True,
            "html_has_sessions_api": True,
            "html_has_preview_api": True,
            "html_has_blocks_api": True,
            "sessions_api_summary": {"total_sessions": 2, "approved_sessions": 2, "trainable_sessions": 2, "exception_sessions": 1},
            "preview_summary": {"eligible_sessions": 2, "eligible_blocks": 8, "fold_count": 1},
            "blocks_api_summary": {"total_sessions": 2, "total_blocks": 8, "trainable_blocks": 8},
            "sessions": [{"id": 1, "status": "closed", "approved_for_training": 1, "data_start_ts": 1.0, "data_end_ts": 2.0, "block_count": 4, "trainable_block_count": 4}],
            "blocks_total": 8,
            "selected_session_ids_api": [1, 2],
            "selected_session_ids_db": [1, 2],
            "selected_block_ids_api": [1, 2],
            "selected_block_ids_db": [1, 2],
            "run_snapshot_session_ids": [1, 2],
            "run_snapshot_ids": [1, 2],
            "sessions_match_db": True,
            "selection_matches_db": True,
            "snapshot_sessions_match": True,
            "snapshot_matches_selection": True,
            "record_count_mismatches": [],
            "max_gap_mismatches": [],
            "runs": [{"id": 1, "status": "completed", "selected_block_count": 2, "sequence_length": 4, "prediction_horizon_sec": 240}],
        }
    )

    assert "# Training Sessions Runtime Audit" in report
    assert "## UI" in report
    assert "## Sessions API Summary" in report
    assert "## Cohort Preview Summary" in report
    assert "## Selection Consistency" in report
    assert "## SQLite Integrity" in report
    assert "Run 1: status=completed" in report
    assert "Session selection matches DB: True" in report
    assert "API selection matches DB: True" in report
    assert "Training snapshot matches selected sessions: True" in report
    assert "Training snapshot matches selected blocks: True" in report
