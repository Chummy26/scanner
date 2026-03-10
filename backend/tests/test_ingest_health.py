from src.spread.ingest_health import ExchangeCircuitBreaker


def test_exchange_circuit_breaker_tracks_window_counts_incrementally():
    breaker = ExchangeCircuitBreaker(window_sec=10.0, threshold_pct=80.0, min_attempts=3, cooldown_sec=30.0)

    breaker.record("mexc", accepted=True, now_ts=0.0)
    breaker.record("mexc", accepted=False, now_ts=1.0)
    state = breaker.record("mexc", accepted=False, now_ts=2.0)

    snapshot = breaker.snapshot(now_ts=2.0)

    assert state == "CLOSED"
    assert snapshot["mexc"]["attempts"] == 3
    assert snapshot["mexc"]["rejections"] == 2
    assert snapshot["mexc"]["rejection_rate_pct"] == (2 / 3) * 100.0


def test_exchange_circuit_breaker_prunes_expired_events_without_recounting():
    breaker = ExchangeCircuitBreaker(window_sec=5.0, threshold_pct=50.0, min_attempts=2, cooldown_sec=30.0)

    breaker.record("xt", accepted=False, now_ts=0.0)
    breaker.record("xt", accepted=False, now_ts=1.0)
    breaker.record("xt", accepted=True, now_ts=7.0)

    snapshot = breaker.snapshot(now_ts=7.0)

    assert snapshot["xt"]["attempts"] == 1
    assert snapshot["xt"]["rejections"] == 0
    assert snapshot["xt"]["rejection_rate_pct"] == 0.0


def test_exchange_circuit_breaker_opens_and_recovers_after_cooldown():
    breaker = ExchangeCircuitBreaker(window_sec=5.0, threshold_pct=50.0, min_attempts=2, cooldown_sec=10.0)

    breaker.record("kucoin", accepted=False, now_ts=0.0)
    state = breaker.record("kucoin", accepted=False, now_ts=1.0)
    assert state == "OPEN"
    assert breaker.is_active("kucoin", now_ts=5.0) is False

    assert breaker.is_active("kucoin", now_ts=12.0) is True
    assert breaker.get_state("kucoin", now_ts=12.0) == "HALF_OPEN"

    breaker.record("kucoin", accepted=True, now_ts=12.0)
    breaker.record("kucoin", accepted=True, now_ts=13.0)
    assert breaker.get_state("kucoin", now_ts=13.0) == "CLOSED"
