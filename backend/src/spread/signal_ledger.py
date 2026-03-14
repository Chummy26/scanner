"""Signal Ledger — persistent record of published ML signals and their outcomes.

Tracks every EXECUTE/STRONG_EXECUTE signal with immutable snapshot data,
then resolves outcomes (hit / miss_timeout) when the horizon expires.
Uses the same SQLite database as the tracker (WAL mode).

ETA is stored as single-point (eta_at_signal). Multi-quantile columns
(eta_q10..q90) will be added via ALTER TABLE when ETA-1 is implemented.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS signal_ledger (
    signal_id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at             REAL DEFAULT (strftime('%s','now')),
    ts_emitted             REAL NOT NULL,
    pair_key               TEXT NOT NULL,
    signal_action          TEXT NOT NULL,
    policy_name            TEXT NOT NULL,
    horizon_sec            INTEGER NOT NULL,
    entry_at_signal        REAL NOT NULL,
    exit_at_signal         REAL NOT NULL,
    target_exit_shallow    REAL,
    target_exit_median     REAL,
    target_exit_deep       REAL,
    target_net_shallow     REAL,
    target_net_median      REAL,
    target_net_deep        REAL,
    prob_at_signal         REAL,
    eta_at_signal          REAL,
    support_2h             INTEGER,
    support_24h            INTEGER,
    context_strength       TEXT,
    range_status           TEXT,
    drift_status           TEXT,
    signal_score           REAL,
    model_version          TEXT,
    feature_contract       TEXT,
    execute_threshold_used REAL,
    default_cost_pct_used  REAL,
    min_net_capture_used   REAL,
    outcome_status         TEXT DEFAULT 'pending',
    outcome_ts             REAL,
    outcome_exit           REAL,
    outcome_total          REAL,
    outcome_duration_sec   REAL,
    best_total_in_horizon  REAL,
    label_hit_shallow      INTEGER,
    label_hit_median       INTEGER,
    label_hit_deep         INTEGER,
    UNIQUE(pair_key, ts_emitted)
)
"""

_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_ledger_pending
    ON signal_ledger(pair_key, outcome_status) WHERE outcome_status = 'pending'
"""


@dataclass(slots=True)
class SignalSnapshot:
    """Immutable snapshot of a published signal."""
    signal_id: int
    ts_emitted: float
    pair_key: str
    signal_action: str
    policy_name: str
    horizon_sec: int
    entry_at_signal: float
    exit_at_signal: float
    prob_at_signal: float
    eta_at_signal: float
    signal_score: float
    outcome_status: str


class SignalLedger:
    """Manages the signal_ledger SQLite table."""

    def __init__(self, db_path: Path | str):
        self.db_path = str(db_path)
        self._ensure_table()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_table(self) -> None:
        with self._connect() as conn:
            conn.execute(_CREATE_TABLE)
            conn.execute(_CREATE_INDEX)

    def check_cooldown(
        self,
        pair_key: str,
        policy_name: str,
        horizon_sec: int,
    ) -> bool:
        """Return True if a pending signal exists (cooldown active)."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT signal_id, ts_emitted FROM signal_ledger "
                "WHERE pair_key = ? AND policy_name = ? AND outcome_status = 'pending' "
                "ORDER BY ts_emitted DESC LIMIT 1",
                (str(pair_key), str(policy_name)),
            ).fetchone()
            if row is None:
                return False
            # Also enforce post-resolution cooldown: horizon_sec / 2
            elapsed = time.time() - float(row["ts_emitted"])
            return elapsed < float(horizon_sec) * 1.5  # horizon + half cooldown

    def record_signal(
        self,
        *,
        pair_key: str,
        signal_action: str,
        policy_name: str,
        horizon_sec: int,
        entry_at_signal: float,
        exit_at_signal: float,
        exit_policies: list[dict[str, Any]],
        prob: float,
        eta_seconds: float,
        support_2h: int,
        support_24h: int,
        context_strength: str,
        range_status: str,
        drift_status: str,
        signal_score: float,
        model_version: str,
        feature_contract: str,
        execute_threshold: float,
        cost_pct: float,
        min_net_capture: float,
    ) -> Optional[int]:
        """Insert a signal if no active cooldown. Returns signal_id or None."""
        if self.check_cooldown(pair_key, policy_name, horizon_sec):
            return None

        # Extract per-policy targets
        targets: dict[str, dict[str, float]] = {}
        for p in exit_policies:
            targets[str(p["name"])] = p

        ts_now = time.time()
        with self._connect() as conn:
            try:
                cursor = conn.execute(
                    """INSERT INTO signal_ledger (
                        ts_emitted, pair_key, signal_action, policy_name, horizon_sec,
                        entry_at_signal, exit_at_signal,
                        target_exit_shallow, target_exit_median, target_exit_deep,
                        target_net_shallow, target_net_median, target_net_deep,
                        prob_at_signal, eta_at_signal,
                        support_2h, support_24h, context_strength, range_status, drift_status,
                        signal_score, model_version, feature_contract,
                        execute_threshold_used, default_cost_pct_used, min_net_capture_used
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        ts_now, str(pair_key), str(signal_action), str(policy_name), int(horizon_sec),
                        float(entry_at_signal), float(exit_at_signal),
                        float(targets.get("shallow", {}).get("exit_target", 0.0)),
                        float(targets.get("median", {}).get("exit_target", 0.0)),
                        float(targets.get("deep", {}).get("exit_target", 0.0)),
                        float(targets.get("shallow", {}).get("net_capture", 0.0)),
                        float(targets.get("median", {}).get("net_capture", 0.0)),
                        float(targets.get("deep", {}).get("net_capture", 0.0)),
                        float(prob), float(eta_seconds),
                        int(support_2h), int(support_24h),
                        str(context_strength), str(range_status), str(drift_status),
                        float(signal_score), str(model_version), str(feature_contract),
                        float(execute_threshold), float(cost_pct), float(min_net_capture),
                    ),
                )
                return cursor.lastrowid
            except sqlite3.IntegrityError:
                logger.debug("Signal dedup: duplicate (pair_key=%s, ts=%.1f)", pair_key, ts_now)
                return None

    def resolve(
        self,
        signal_id: int,
        *,
        hit: bool,
        actual_exit: float,
        actual_total: float,
        duration_sec: float,
        best_total_in_horizon: float,
        target_net_shallow: float,
        target_net_median: float,
        target_net_deep: float,
        cost_pct: float,
    ) -> None:
        """Mark a pending signal as hit or miss_timeout."""
        best_net = float(best_total_in_horizon) - float(cost_pct)
        label_shallow = 1 if best_net >= float(target_net_shallow) else 0
        label_median = 1 if best_net >= float(target_net_median) else 0
        label_deep = 1 if best_net >= float(target_net_deep) else 0

        with self._connect() as conn:
            conn.execute(
                """UPDATE signal_ledger SET
                    outcome_status = ?, outcome_ts = ?, outcome_exit = ?, outcome_total = ?,
                    outcome_duration_sec = ?, best_total_in_horizon = ?,
                    label_hit_shallow = ?, label_hit_median = ?, label_hit_deep = ?
                WHERE signal_id = ? AND outcome_status = 'pending'""",
                (
                    "hit" if hit else "miss_timeout",
                    time.time(),
                    float(actual_exit), float(actual_total), float(duration_sec),
                    float(best_total_in_horizon),
                    label_shallow, label_median, label_deep,
                    int(signal_id),
                ),
            )

    def pending_signals(self) -> list[SignalSnapshot]:
        """Return all pending signals for resolution checking."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT signal_id, ts_emitted, pair_key, signal_action, policy_name, "
                "horizon_sec, entry_at_signal, exit_at_signal, prob_at_signal, "
                "eta_at_signal, signal_score, outcome_status "
                "FROM signal_ledger WHERE outcome_status = 'pending' "
                "ORDER BY ts_emitted ASC",
            ).fetchall()
        return [
            SignalSnapshot(
                signal_id=int(r["signal_id"]),
                ts_emitted=float(r["ts_emitted"]),
                pair_key=str(r["pair_key"]),
                signal_action=str(r["signal_action"]),
                policy_name=str(r["policy_name"]),
                horizon_sec=int(r["horizon_sec"]),
                entry_at_signal=float(r["entry_at_signal"]),
                exit_at_signal=float(r["exit_at_signal"]),
                prob_at_signal=float(r["prob_at_signal"]),
                eta_at_signal=float(r["eta_at_signal"]),
                signal_score=float(r["signal_score"]),
                outcome_status=str(r["outcome_status"]),
            )
            for r in rows
        ]

    def signal_count(self, *, status: str | None = None) -> int:
        """Count signals, optionally filtered by status."""
        with self._connect() as conn:
            if status:
                row = conn.execute(
                    "SELECT COUNT(*) FROM signal_ledger WHERE outcome_status = ?",
                    (str(status),),
                ).fetchone()
            else:
                row = conn.execute("SELECT COUNT(*) FROM signal_ledger").fetchone()
            return int(row[0]) if row else 0


__all__ = ["SignalLedger", "SignalSnapshot"]
