"""Token-bucket rate limiting (client side) and the shared ``TokenBucket``.

Why token buckets: the provider's own ``x-ratelimit-reset-*`` headers show it
refills continuously (e.g. reset-tokens == tokens_used / (TPM/60)), so a bucket
with the same capacity and refill rate predicts the provider's decisions closely
while still allowing short bursts. Fixed windows would permit 2x bursts at window
edges and sliding-window logs cost memory per request for no extra fidelity.
"""
from __future__ import annotations

import asyncio
import time
from typing import Callable


class TokenBucket:
    """Continuously refilling bucket. ``refill_per_s == 0`` means a hard budget."""

    def __init__(self, capacity: float, refill_per_s: float, clock: Callable[[], float] = time.monotonic):
        self.capacity = float(capacity)
        self.refill_per_s = float(refill_per_s)
        self._clock = clock
        self._level = float(capacity)
        self._t = clock()

    def _refill(self) -> None:
        now = self._clock()
        self._level = min(self.capacity, self._level + (now - self._t) * self.refill_per_s)
        self._t = now

    @property
    def level(self) -> float:
        self._refill()
        return self._level

    def wait_for(self, n: float) -> float:
        """Seconds until ``n`` units are available (0 if now, inf if never)."""
        self._refill()
        if n > self.capacity:
            return float("inf")
        if self._level >= n:
            return 0.0
        if self.refill_per_s <= 0:
            return float("inf")
        return (n - self._level) / self.refill_per_s

    def take(self, n: float) -> None:
        self._refill()
        self._level -= n

    def give(self, n: float) -> None:
        self._refill()
        self._level = min(self.capacity, self._level + n)

    def set_level(self, n: float) -> None:
        self._refill()
        self._level = max(0.0, min(self.capacity, n))

    def reset_after_s(self) -> float:
        """Seconds until the bucket is full again."""
        self._refill()
        if self.refill_per_s <= 0:
            return float("inf")
        return (self.capacity - self._level) / self.refill_per_s


class LimiterTimeout(Exception):
    """Raised when the limiter cannot grant capacity before the deadline."""

    def __init__(self, needed_wait_s: float) -> None:
        super().__init__(f"would need to wait {needed_wait_s:.1f}s")
        self.needed_wait_s = needed_wait_s


class ClientLimiter:
    """RPM + TPM buckets that mirror the provider, plus daily-budget state.

    ``acquire`` is FIFO (an asyncio lock is held while waiting) so waiters are
    served in arrival order and do not stampede when capacity returns.
    """

    def __init__(self, rpm: int, tpm: int, daily_budget: int, enabled: bool = True) -> None:
        self.enabled = enabled
        self.rpm_base_rate = rpm / 60
        self.rate_factor = 1.0      # AIMD multiplier on the RPM refill rate
        self.rpm = TokenBucket(rpm, rpm / 60)
        self.tpm = TokenBucket(tpm, tpm / 60)
        self.tpm_capacity = tpm
        self.daily_budget = daily_budget
        self.daily_used = 0
        self.daily_remaining_hint: int | None = None   # from x-ratelimit-remaining-requests
        self._lock = asyncio.Lock()
        self._exhausted_until: float = 0.0   # monotonic; calls blocked until then
        self._exhausted = False

    # -- daily budget -------------------------------------------------------
    def mark_daily_exhausted(self, probe_after_s: float) -> None:
        """Daily quota is gone: block API calls, allow one probe after ``probe_after_s``."""
        self._exhausted = True
        self._exhausted_until = time.monotonic() + probe_after_s

    def clear_daily_exhausted(self) -> None:
        self._exhausted = False

    @property
    def daily_exhausted(self) -> bool:
        return self._exhausted and time.monotonic() < self._exhausted_until

    @property
    def probe_due(self) -> bool:
        return self._exhausted and time.monotonic() >= self._exhausted_until

    def seconds_until_probe(self) -> float:
        return max(0.0, self._exhausted_until - time.monotonic())

    def daily_remaining(self) -> int:
        if self.daily_remaining_hint is not None:
            return self.daily_remaining_hint
        return max(0, self.daily_budget - self.daily_used)

    # -- per-minute buckets -------------------------------------------------
    async def acquire(self, tokens: int, max_wait_s: float) -> float:
        """Wait for 1 request + ``tokens`` TPM capacity. Returns seconds waited.

        Raises ``LimiterTimeout`` immediately if the wait would exceed ``max_wait_s``.
        """
        if not self.enabled:
            return 0.0
        t0 = time.monotonic()
        async with self._lock:
            tokens = min(tokens, int(self.tpm_capacity))  # an oversize request can never fit; let the provider decide
            while True:
                wait = max(self.rpm.wait_for(1), self.tpm.wait_for(tokens))
                if wait <= 0:
                    self.rpm.take(1)
                    self.tpm.take(tokens)
                    self.daily_used += 1
                    return time.monotonic() - t0
                elapsed = time.monotonic() - t0
                if elapsed + wait > max_wait_s:
                    raise LimiterTimeout(wait)
                await asyncio.sleep(min(wait, 1.0))

    # -- adaptation (AIMD) ----------------------------------------------------
    def on_rate_limited(self, kind: str) -> None:
        """A 429 means our model of the provider's limit is too optimistic: halve the RPM rate
        (multiplicative decrease, floor 20%) and empty the RPM bucket for a request-rate 429."""
        if not self.enabled:
            return
        if kind == "rpm":
            self.rate_factor = max(0.2, self.rate_factor * 0.5)
            self.rpm.refill_per_s = self.rpm_base_rate * self.rate_factor
            self.rpm.set_level(0.0)

    def on_success(self) -> None:
        """Additive recovery (+2% of the configured rate per success) toward the configured limit."""
        if self.enabled and self.rate_factor < 1.0:
            self.rate_factor = min(1.0, self.rate_factor + 0.02)
            self.rpm.refill_per_s = self.rpm_base_rate * self.rate_factor

    def refund_tokens(self, n: int) -> None:
        if self.enabled and n > 0:
            self.tpm.give(n)

    def sync_from_headers(self, headers: dict[str, str]) -> None:
        """Trust the provider's view of remaining capacity when it is lower than ours."""
        if not self.enabled:
            return
        try:
            rem_t = headers.get("x-ratelimit-remaining-tokens")
            if rem_t is not None and float(rem_t) < self.tpm.level:
                self.tpm.set_level(float(rem_t))
            rem_r = headers.get("x-ratelimit-remaining-requests")
            if rem_r is not None:
                self.daily_remaining_hint = int(float(rem_r))
        except ValueError:
            pass

    def snapshot(self) -> dict[str, float | int | bool]:
        return {
            "tpm_available": int(self.tpm.level),
            "rpm_available": int(self.rpm.level),
            "daily_remaining": self.daily_remaining(),
            "daily_exhausted": self.daily_exhausted,
        }
