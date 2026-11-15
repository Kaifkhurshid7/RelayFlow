"""Stand-in provider configuration and rate-limit state.

Limits are modelled as continuously refilling token buckets because that is what
the real service's ``x-ratelimit-reset-*`` headers imply (see
``experiments/observations``). Everything is configurable via ``MOCK_*``
environment variables or live through ``POST /admin/config``.

This is an approximation of observable behaviour. It does not reproduce Groq's
internal infrastructure.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

from pydantic_settings import BaseSettings, SettingsConfigDict

from app.scheduler.rate_limiter import TokenBucket


class MockConfig(BaseSettings):
    """All stand-in knobs. Defaults mirror the documented free-tier limits for
    openai/gpt-oss-120b; latency defaults are overwritten by calibration."""

    model_config = SettingsConfigDict(env_prefix="MOCK_", extra="ignore")

    # latency
    base_latency_ms: float = 500.0       # time to first token at zero load
    latency_jitter_ms: float = 200.0     # std-dev of gaussian jitter
    prefill_ms_per_1k_tokens: float = 0.0
    tokens_per_s: float = 300.0          # generation speed after first token
    reasoning_tokens: int = 20           # hidden reasoning emitted before content
    # limits
    max_rpm: int = 30
    max_tpm: int = 8_000
    max_daily_requests: int = 1_000
    daily_refill_per_s: float = 1000 / 86_400  # observed: 86.4 s per request (inferred)
    max_concurrency: int = 10
    concurrency_mode: str = "reject"     # "reject" -> 429, "queue" -> wait
    # faults
    failure_rate: float = 0.02           # HTTP 500
    slow_rate: float = 0.0               # fraction of requests made slow
    slow_factor: float = 5.0
    hang_rate: float = 0.0               # fraction that never answers in time (client timeout)
    hang_s: float = 120.0
    partial_rate: float = 0.0            # stream is cut mid-response
    # load behaviour
    overload_factor: float = 1.5         # latency multiplier at full concurrency
    pressure_window_s: float = 10.0
    pressure_weight: float = 0.5         # how strongly arrivals above capacity slow service
    rejected_cost_load: bool = True      # rejected requests still burden the provider
    # misc
    streaming_enabled: bool = True       # False -> whole answer delivered at once
    prompt_overhead_tokens: int = 70     # measured: "Say ok" = 73 prompt tokens on Groq
    tokens_per_word: float = 1.5
    seed: int = 1234


def fmt_duration(s: float) -> str:
    """Format seconds the way Groq does: '960ms', '3.465s', '1m26.4s'."""
    if s != s or s == float("inf"):
        return "never"
    if s < 1:
        return f"{round(s * 1000)}ms"
    if s < 60:
        return f"{s:.3f}".rstrip("0").rstrip(".") + "s"
    m, sec = divmod(s, 60)
    return f"{int(m)}m{sec:.1f}s".replace(".0s", "s")


@dataclass
class Counters:
    arrivals: int = 0
    accepted: int = 0
    rejected_rpm: int = 0
    rejected_tpm: int = 0
    rejected_daily: int = 0
    rejected_concurrency: int = 0
    rejected_too_large: int = 0
    errors_500: int = 0
    hangs: int = 0
    partial: int = 0
    completed: int = 0
    peak_in_flight: int = 0
    tokens_served: int = 0


@dataclass
class LimitState:
    """Mutable provider state: buckets, in-flight count and arrival history."""

    cfg: MockConfig
    clock: Callable[[], float] = time.monotonic
    in_flight: int = 0
    counters: Counters = field(default_factory=Counters)
    _arrivals: deque = field(default_factory=deque)

    def __post_init__(self) -> None:
        self.rebuild()

    def rebuild(self) -> None:
        c = self.cfg
        self.rpm = TokenBucket(c.max_rpm, c.max_rpm / 60, self.clock)
        self.tpm = TokenBucket(c.max_tpm, c.max_tpm / 60, self.clock)
        self.rpd = TokenBucket(c.max_daily_requests, c.daily_refill_per_s, self.clock)

    def reset(self) -> None:
        self.in_flight = 0
        self.counters = Counters()
        self._arrivals.clear()
        self.rebuild()

    def note_arrival(self) -> None:
        self._arrivals.append(self.clock())

    def arrival_pressure(self) -> float:
        """Recent arrival rate relative to the RPM capacity (1.0 = at capacity)."""
        now, w = self.clock(), self.cfg.pressure_window_s
        while self._arrivals and now - self._arrivals[0] > w:
            self._arrivals.popleft()
        return (len(self._arrivals) * 60.0 / w) / max(1, self.cfg.max_rpm)

    def check(self, tokens_needed: float) -> tuple[str, float] | None:
        """Return (limit_name, wait_s) for the first limit that would be exceeded."""
        w = self.rpd.wait_for(1)
        if w > 0:
            return "rpd", w
        w = self.rpm.wait_for(1)
        if w > 0:
            return "rpm", w
        w = self.tpm.wait_for(tokens_needed)
        if w > 0:
            return "tpm", w
        return None

    def rate_headers(self) -> dict[str, str]:
        return {
            "x-ratelimit-limit-requests": str(self.cfg.max_daily_requests),
            "x-ratelimit-limit-tokens": str(self.cfg.max_tpm),
            "x-ratelimit-remaining-requests": str(max(0, int(self.rpd.level))),
            "x-ratelimit-remaining-tokens": str(max(0, int(self.tpm.level))),
            "x-ratelimit-reset-requests": fmt_duration(self.rpd.reset_after_s()),
            "x-ratelimit-reset-tokens": fmt_duration(self.tpm.reset_after_s()),
        }
