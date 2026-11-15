"""Stand-in provider behaviour + provider client + gateway (rate limits, retries,
streaming, timeouts, partial responses, daily quota)."""
import asyncio
import time

import pytest

from app.config import SchedulerSettings
from app.providers.base import (
    BAD_REQUEST, CONNECTION, PARTIAL, RATE_DAILY, RATE_MINUTE, SERVER, TIMEOUT, ProviderResult,
    classify_429, classify_status, parse_retry_hint,
)
from app.providers.mock_provider import MockProvider
from app.scheduler.gateway import ProviderGateway
from app.scheduler.rate_limiter import ClientLimiter, LimiterTimeout, TokenBucket
from app.scheduler.retry import RetryBudget, RetryPolicy

MSGS = [
    {"role": "system", "content": "sys"},
    {"role": "user", "content": "Sources:\n\n[S1] (Attention Is All You Need | arXiv:1706.03762 v7 | 3.2.2 Multi-Head Attention | p.4)\n"
     "In this work we employ h = 8 parallel attention layers, or heads. For each of these we use dk = dv = dmodel/h = 64.\n\n"
     "Question: How many parallel attention heads does the Transformer use?"},
]


def prov(base: str) -> MockProvider:
    return MockProvider(base, "mock")


# ---- pure unit tests ---------------------------------------------------------
def test_token_bucket_refills_and_waits() -> None:
    t = [0.0]
    b = TokenBucket(10, 2.0, clock=lambda: t[0])
    b.take(10)
    assert b.wait_for(4) == pytest.approx(2.0)
    t[0] = 3.0
    assert b.level == pytest.approx(6.0)
    assert b.wait_for(11) == float("inf")  # more than capacity can never be granted


def test_error_classification() -> None:
    assert classify_429("... on tokens per minute (TPM): Limit 8000 ...") == RATE_MINUTE
    assert classify_429("... on requests per day (RPD): Limit 1000 ...") == RATE_DAILY
    assert classify_status(500) == SERVER and classify_status(401) != RATE_MINUTE
    assert classify_status(413) == BAD_REQUEST
    assert parse_retry_hint("Please try again in 1m26.4s.") == pytest.approx(86.4)  # compound form
    assert parse_retry_hint("Please try again in 3.2s.") == pytest.approx(3.2)
    assert parse_retry_hint("Please try again in 250ms") == pytest.approx(0.25)


def test_retry_policy_decisions() -> None:
    cfg = SchedulerSettings(max_retries=2, backoff_base_s=1.0, backoff_max_s=4.0, jitter=False)
    pol = RetryPolicy(cfg, seed=1)
    r429 = ProviderResult(ok=False, http_status=429, error_type=RATE_MINUTE, retry_after_s=3.0)
    d = pol.decide(r429, 1, time_left_s=60)
    assert d.retry and d.wait_s == 3.0                      # Retry-After beats the 1s backoff
    assert pol.decide(r429, 3, 60).reason == "max_retries_reached"
    assert not pol.decide(ProviderResult(ok=False, error_type=RATE_DAILY), 1, 60).retry   # daily: never
    assert not pol.decide(ProviderResult(ok=False, error_type=BAD_REQUEST), 1, 60).retry   # permanent
    assert not pol.decide(r429, 1, time_left_s=2.0).retry   # would blow the deadline
    assert pol.decide(ProviderResult(ok=False, error_type=SERVER), 2, 60).wait_s == 2.0     # exponential


def test_backoff_jitter_is_bounded_and_spread() -> None:
    pol = RetryPolicy(SchedulerSettings(backoff_base_s=1.0, backoff_max_s=8.0, jitter=True), seed=7)
    waits = [pol.backoff(3) for _ in range(200)]
    assert all(0 <= w <= 4.0 for w in waits) and len({round(w, 2) for w in waits}) > 50


def test_retry_budget_blocks_storms() -> None:
    b = RetryBudget(ratio=0.5, floor=0)
    for _ in range(10):
        b.note_first_attempt()
    allowed = 0
    while b.allow():
        b.note_retry()
        allowed += 1
    assert allowed == 5


async def test_client_limiter_waits_then_times_out() -> None:
    lim = ClientLimiter(rpm=600, tpm=600, daily_budget=100)   # 10 tokens/s
    await lim.acquire(600, 5)                                   # drains the bucket
    waited = await lim.acquire(10, 5)                           # needs ~1s of refill
    assert 0.5 < waited < 2.0
    with pytest.raises(LimiterTimeout):
        await lim.acquire(600, 1.0)                             # would need ~60s


# ---- stand-in + client -----------------------------------------------------------
async def test_streaming_first_token_before_end_and_usage(mock) -> None:
    base = mock(base_latency_ms=200, tokens_per_s=50)
    p = prov(base)
    r = await p.generate(MSGS, max_tokens=300, timeout_s=10, request_key="s1")
    await p.aclose()
    assert r.ok and r.http_status == 200 and "[S1]" in r.text
    assert r.first_reasoning_ms is not None and r.first_reasoning_ms < r.first_token_ms < r.total_ms
    assert r.first_token_ms >= 180            # ~base latency
    assert r.tokens_in and r.tokens_out and r.reasoning_tokens is not None
    assert "x-ratelimit-remaining-tokens" in r.headers


async def test_non_streaming_mode_delivers_everything_at_the_end(mock) -> None:
    base = mock(base_latency_ms=100, streaming_enabled=False, tokens_per_s=100)
    p = prov(base)
    r = await p.generate(MSGS, max_tokens=300, timeout_s=10)
    await p.aclose()
    assert r.ok and r.total_ms - r.first_token_ms < 150   # first content ~= end


async def test_rpm_limit_returns_429_with_retry_after(mock) -> None:
    base = mock(max_rpm=2)
    p = prov(base)
    rs = [await p.generate(MSGS, max_tokens=100, timeout_s=5) for _ in range(3)]
    await p.aclose()
    assert [r.ok for r in rs] == [True, True, False]
    r = rs[2]
    assert r.http_status == 429 and r.error_type == RATE_MINUTE and r.retry_after_s and r.retry_after_s >= 1
    assert "retry-after" in r.headers
    assert mock.stats()["rejected_rpm"] == 1


async def test_tpm_limit_and_oversize(mock) -> None:
    base = mock(max_tpm=1000, base_latency_ms=300)
    p = prov(base)
    # Two requests in flight at once: each reserves prompt+600 tokens, so the bucket (1000) cannot hold both.
    # (Unused max_tokens are only refunded once a request completes, as on the real service.)
    a, b2 = await asyncio.gather(p.generate(MSGS, max_tokens=600, timeout_s=5),
                                 p.generate(MSGS, max_tokens=600, timeout_s=5))
    huge = await p.generate(MSGS, max_tokens=5000, timeout_s=5)        # > bucket capacity -> 413, never retryable
    await p.aclose()
    first, second = sorted([a, b2], key=lambda r: r.ok, reverse=True)
    assert first.ok and second.error_type == RATE_MINUTE and "tokens per minute" in second.error_message
    assert second.retry_after_s and second.retry_after_s > 0
    assert huge.http_status == 413 and huge.error_type == BAD_REQUEST


async def test_concurrency_limit_rejects(mock) -> None:
    base = mock(max_concurrency=2, base_latency_ms=400)
    p = prov(base)
    rs = await asyncio.gather(*[p.generate(MSGS, max_tokens=100, timeout_s=10) for _ in range(5)])
    await p.aclose()
    assert sum(r.ok for r in rs) == 2 and sum(r.http_status == 429 for r in rs) == 3


async def test_http_500_and_reproducible_failures(mock) -> None:
    base = mock(failure_rate=0.5)
    p = prov(base)
    first = [(await p.generate(MSGS, max_tokens=100, timeout_s=5, request_key=f"k{i}")).http_status for i in range(12)]
    mock()  # reset counters/seed state
    mock(failure_rate=0.5)
    second = [(await p.generate(MSGS, max_tokens=100, timeout_s=5, request_key=f"k{i}")).http_status for i in range(12)]
    await p.aclose()
    assert 500 in first and 200 in first
    assert first == second            # same keys -> same faults, independent of arrival order


async def test_timeout_and_partial_response(mock) -> None:
    base = mock(hang_rate=1.0, hang_s=30)
    p = prov(base)
    t0 = time.perf_counter()
    r = await p.generate(MSGS, max_tokens=100, timeout_s=1.0)
    assert r.error_type == TIMEOUT and not r.ok and time.perf_counter() - t0 < 3
    base = mock(partial_rate=1.0, tokens_per_s=20)
    r2 = await p.generate(MSGS, max_tokens=100, timeout_s=10)
    await p.aclose()
    assert not r2.ok and r2.error_type in (PARTIAL, CONNECTION) and r2.first_token_ms is not None or r2.error_type == CONNECTION


async def test_load_makes_latency_grow(mock) -> None:
    base = mock(base_latency_ms=100, overload_factor=4.0, max_concurrency=8, tokens_per_s=2000)
    p = prov(base)
    solo = await p.generate(MSGS, max_tokens=100, timeout_s=10)
    rs = await asyncio.gather(*[p.generate(MSGS, max_tokens=100, timeout_s=10) for _ in range(8)])
    await p.aclose()
    assert max(r.first_token_ms for r in rs if r.ok) > 2 * solo.first_token_ms


# ---- gateway -------------------------------------------------------------------------
def gw(base: str, **kw) -> ProviderGateway:
    cfg = SchedulerSettings(use_client_limiter=False, backoff_base_s=0.05, backoff_max_s=0.3, **kw)
    return ProviderGateway(prov(base), cfg, seed=3)


async def test_gateway_retries_500_then_succeeds(mock) -> None:
    base = mock(failure_rate=0.6)
    g = gw(base, max_retries=8, retry_budget_ratio=1000.0)   # budget off: isolate backoff + retry
    res = [await g.call(MSGS, max_tokens=100, deadline_mono=time.monotonic() + 20, request_id=f"r{i}") for i in range(6)]
    assert all(r.ok for r in res) and sum(r.retries for r in res) > 0
    assert all(a.http_status in (200, 500) for r in res for a in r.attempts)


async def test_retry_budget_stops_retry_storm(mock) -> None:
    base = mock(failure_rate=0.9)
    g = gw(base, max_retries=8, retry_budget_ratio=0.5, retry_budget_floor=2)
    res = [await g.call(MSGS, max_tokens=100, deadline_mono=time.monotonic() + 20, request_id=f"s{i}") for i in range(10)]
    total_retries = sum(r.retries for r in res)
    assert total_retries <= 0.5 * 10 + 2 + 1                      # bounded by the budget, not by max_retries
    assert any(r.terminal_reason == "retry_budget_exhausted" for r in res)


async def test_gateway_honors_retry_after_for_minute_limit(mock) -> None:
    base = mock(max_rpm=30, base_latency_ms=20)     # bucket 30, refills 0.5 requests/s -> Retry-After ~2s
    g = gw(base, max_retries=3)
    await asyncio.gather(*[g.provider.generate(MSGS, max_tokens=50, timeout_s=5) for _ in range(30)])
    r = await g.call(MSGS, max_tokens=50, deadline_mono=time.monotonic() + 20, request_id="x")
    assert r.ok and r.retries >= 1
    assert r.attempts[0].http_status == 429 and r.attempts[0].retry_after_s >= 1
    assert r.retry_wait_s >= r.attempts[0].retry_after_s       # waited at least what the provider asked


async def test_gateway_does_not_retry_daily_exhaustion(mock) -> None:
    base = mock(max_daily_requests=1, daily_refill_per_s=0.0)
    g = gw(base, max_retries=5)
    ok = await g.call(MSGS, max_tokens=50, deadline_mono=time.monotonic() + 10, request_id="a")
    bad = await g.call(MSGS, max_tokens=50, deadline_mono=time.monotonic() + 10, request_id="b")
    blocked = await g.call(MSGS, max_tokens=50, deadline_mono=time.monotonic() + 10, request_id="c")
    assert ok.ok
    assert not bad.ok and bad.terminal_reason == "daily_quota_exhausted" and bad.retries == 0 and len(bad.attempts) == 1
    assert blocked.terminal_reason == "daily_quota_exhausted" and blocked.attempts == []   # no API call at all
    assert mock.stats()["rejected_daily"] == 1


async def test_gateway_respects_deadline(mock) -> None:
    base = mock(hang_rate=1.0, hang_s=30)
    g = gw(base, max_retries=10, attempt_timeout_s=0.5)
    t0 = time.monotonic()
    r = await g.call(MSGS, max_tokens=50, deadline_mono=t0 + 2.0, request_id="d")
    assert not r.ok and time.monotonic() - t0 < 3.5
    assert r.terminal_reason in ("deadline_exceeded", "deadline_would_be_exceeded", "max_retries_reached")


async def test_gateway_waits_for_client_limiter_instead_of_hitting_provider(mock) -> None:
    base = mock(max_rpm=600, base_latency_ms=300)
    cfg = SchedulerSettings(use_client_limiter=True, rpm=600, tpm=1200, daily_budget=100)
    g = ProviderGateway(prov(base), cfg, seed=1)
    r1, r2 = await asyncio.gather(
        g.call(MSGS, max_tokens=500, deadline_mono=time.monotonic() + 30, request_id="1"),
        g.call(MSGS, max_tokens=500, deadline_mono=time.monotonic() + 30, request_id="2"),
    )
    assert r1.ok and r2.ok
    assert max(r1.limiter_wait_s, r2.limiter_wait_s) > 0.5     # one had to wait for TPM capacity
    assert mock.stats()["rejected_tpm"] == 0                    # ...so the provider never had to refuse it


async def test_limiter_adapts_down_on_429_and_recovers() -> None:
    lim = ClientLimiter(rpm=60, tpm=100_000, daily_budget=100)
    lim.on_rate_limited("rpm")
    assert lim.rate_factor == 0.5 and lim.rpm.refill_per_s == pytest.approx(0.5) and lim.rpm.level < 1
    lim.on_rate_limited("rpm"); lim.on_rate_limited("rpm"); lim.on_rate_limited("rpm")
    assert lim.rate_factor == 0.2                              # floor
    lim.on_rate_limited("tpm")                                 # token-limit 429s do not touch the request rate
    assert lim.rate_factor == 0.2
    for _ in range(60):
        lim.on_success()
    assert lim.rate_factor == 1.0 and lim.rpm.refill_per_s == pytest.approx(1.0)
