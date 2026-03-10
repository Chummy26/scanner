from __future__ import annotations

import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class HourlyHealthDigest:
    hour_start_ts: float
    hour_end_ts: float
    records_total: int
    records_rejected: int
    rejection_rate_pct: float
    exchanges_active: int
    exchanges_circuit_open: list[str]
    pairs_with_records: int
    pairs_with_gaps: int
    cross_exchange_flags: int
    avg_book_age_sec: dict[str, float]
    episode_count: int
    quality_verdict: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ExchangeCircuitBreaker:
    def __init__(
        self,
        threshold_pct: float = 80.0,
        window_sec: float = 300.0,
        cooldown_sec: float = 600.0,
        min_attempts: int = 10,
    ):
        self.threshold_pct = float(threshold_pct)
        self.window_sec = float(window_sec)
        self.cooldown_sec = float(cooldown_sec)
        self.min_attempts = max(int(min_attempts), 1)
        self._states: dict[str, str] = {}
        self._events: dict[str, deque[tuple[float, bool]]] = {}
        self._opened_at: dict[str, float] = {}
        self._attempt_counts: dict[str, int] = {}
        self._rejection_counts: dict[str, int] = {}

    def _prune(self, exchange: str, now_ts: float) -> deque[tuple[float, bool]]:
        history = self._events.setdefault(str(exchange), deque())
        cutoff = float(now_ts) - self.window_sec
        while history and history[0][0] < cutoff:
            _, accepted = history.popleft()
            self._attempt_counts[exchange] = max(int(self._attempt_counts.get(exchange, 0) or 0) - 1, 0)
            if not accepted:
                self._rejection_counts[exchange] = max(int(self._rejection_counts.get(exchange, 0) or 0) - 1, 0)
        return history

    def _metrics(self, exchange: str, now_ts: float) -> tuple[int, int, float]:
        self._prune(exchange, now_ts)
        attempts = int(self._attempt_counts.get(exchange, 0) or 0)
        rejections = int(self._rejection_counts.get(exchange, 0) or 0)
        rejection_rate_pct = (float(rejections) / float(attempts) * 100.0) if attempts > 0 else 0.0
        return attempts, rejections, rejection_rate_pct

    def record(self, exchange: str, *, accepted: bool, now_ts: float) -> str:
        exchange = str(exchange or "").strip().lower()
        if not exchange:
            return "CLOSED"
        history = self._prune(exchange, now_ts)
        history.append((float(now_ts), bool(accepted)))
        self._attempt_counts[exchange] = int(self._attempt_counts.get(exchange, 0) or 0) + 1
        if not accepted:
            self._rejection_counts[exchange] = int(self._rejection_counts.get(exchange, 0) or 0) + 1
        else:
            self._rejection_counts.setdefault(exchange, int(self._rejection_counts.get(exchange, 0) or 0))
        state = self._states.get(exchange, "CLOSED")
        attempts, _, rejection_rate_pct = self._metrics(exchange, now_ts)
        if state == "OPEN":
            if (float(now_ts) - float(self._opened_at.get(exchange, 0.0))) >= self.cooldown_sec:
                state = "HALF_OPEN"
                self._states[exchange] = state
        if state == "CLOSED":
            if attempts >= self.min_attempts and rejection_rate_pct > self.threshold_pct:
                state = "OPEN"
                self._states[exchange] = state
                self._opened_at[exchange] = float(now_ts)
        elif state == "HALF_OPEN":
            if attempts >= self.min_attempts and rejection_rate_pct > self.threshold_pct:
                state = "OPEN"
                self._states[exchange] = state
                self._opened_at[exchange] = float(now_ts)
            elif attempts >= self.min_attempts and rejection_rate_pct <= (self.threshold_pct * 0.5):
                state = "CLOSED"
                self._states[exchange] = state
                self._opened_at.pop(exchange, None)
        else:
            self._states.setdefault(exchange, state)
        return self._states.get(exchange, state)

    def is_active(self, exchange: str, now_ts: float | None = None) -> bool:
        exchange = str(exchange or "").strip().lower()
        if not exchange:
            return True
        current_ts = float(now_ts if now_ts is not None else time.time())
        state = self._states.get(exchange, "CLOSED")
        if state == "OPEN" and (current_ts - float(self._opened_at.get(exchange, 0.0))) >= self.cooldown_sec:
            self._states[exchange] = "HALF_OPEN"
            state = "HALF_OPEN"
        return state != "OPEN"

    def get_state(self, exchange: str, now_ts: float | None = None) -> str:
        self.is_active(exchange, now_ts=now_ts)
        return self._states.get(str(exchange or "").strip().lower(), "CLOSED")

    def snapshot(self, *, now_ts: float | None = None) -> dict[str, dict[str, float | int | str]]:
        current_ts = float(now_ts if now_ts is not None else time.time())
        exchanges = set(self._states) | set(self._events)
        payload: dict[str, dict[str, float | int | str]] = {}
        for exchange in sorted(exchanges):
            state = self.get_state(exchange, now_ts=current_ts)
            attempts, rejections, rejection_rate_pct = self._metrics(exchange, current_ts)
            payload[exchange] = {
                "state": state,
                "attempts": int(attempts),
                "rejections": int(rejections),
                "rejection_rate_pct": float(rejection_rate_pct),
            }
        return payload

    def open_exchanges(self, *, now_ts: float | None = None) -> list[str]:
        current_ts = float(now_ts if now_ts is not None else time.time())
        return sorted(
            exchange
            for exchange in set(self._states) | set(self._events)
            if self.get_state(exchange, now_ts=current_ts) == "OPEN"
        )
