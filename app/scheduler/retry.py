"""Bounded retry policy: classification, exponential backoff, jitter, Retry-After,
a per-request deadline, and a global retry budget that prevents retry storms."""
from __future__ import annotations

import random
import time
from collections import deque
from dataclasses import dataclass

from app.config import SchedulerSettings
from app.providers.base import RATE_DAILY, RETRYABLE, ProviderResult


@dataclass
class RetryDecision:
    retry: bool
    wait_s: float
    reason: str


class RetryBudget:
    """Cap retries to a fraction of recent first attempts (plus a small floor).

    Without it, a provider brownout turns every failed request into several more,
    multiplying load exactly when the provider can least absorb it.
    """

    def __init__(self, ratio: float, floor: int, window_s: float = 30.0) -> None:
        self.ratio, self.floor, self.window_s = ratio, floor, window_s
        self._first: deque[float] = deque()
        self._retries: deque[float] = deque()

    def _trim(self) -> None:
        cut = time.monotonic() - self.window_s
        for q in (self._first, self._retries):
            while q and q[0] < cut:
                q.popleft()

    def note_first_attempt(self) -> None:
        self._first.append(time.monotonic())

    def allow(self) -> bool:
        self._trim()
        return len(self._retries) < self.ratio * len(self._first) + self.floor

    def note_retry(self) -> None:
        self._retries.append(time.monotonic())


class RetryPolicy:
    """Decides whether and how long to wait before the next attempt."""

    def __init__(self, cfg: SchedulerSettings, budget: RetryBudget | None = None, seed: int | None = None):
        self.cfg = cfg
        self.budget = budget or RetryBudget(cfg.retry_budget_ratio, cfg.retry_budget_floor)
        self._rng = random.Random(seed)

    def backoff(self, attempt: int) -> float:
        """Exponential backoff for the retry after ``attempt`` (1-based), with optional full jitter."""
        exp = min(self.cfg.backoff_max_s, self.cfg.backoff_base_s * 2 ** (attempt - 1))
        return self._rng.uniform(0.0, exp) if self.cfg.jitter else exp

    def decide(self, res: ProviderResult, attempt: int, time_left_s: float) -> RetryDecision:
        et = res.error_type or "unknown_error"
        if et == RATE_DAILY:
            return RetryDecision(False, 0.0, "daily_quota_exhausted_not_retryable")
        if et not in RETRYABLE:
            return RetryDecision(False, 0.0, f"{et}_not_retryable")
        if attempt > self.cfg.max_retries:
            return RetryDecision(False, 0.0, "max_retries_reached")
        wait = self.backoff(attempt)
        if self.cfg.honor_retry_after and res.retry_after_s:
            # small jitter on top so waiters released by the same header do not collide
            wait = max(wait, res.retry_after_s + (self._rng.uniform(0, 0.25) if self.cfg.jitter else 0.0))
        if wait >= time_left_s:
            return RetryDecision(False, wait, "deadline_would_be_exceeded")
        if not self.budget.allow():
            return RetryDecision(False, wait, "retry_budget_exhausted")
        return RetryDecision(True, wait, et)
