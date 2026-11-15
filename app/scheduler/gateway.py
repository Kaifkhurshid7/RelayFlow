"""ProviderGateway: every LLM call goes through here.

Responsibilities, in order, for each attempt:
  1. refuse to call at all if the daily quota is known to be exhausted
  2. wait for client-side RPM/TPM capacity (token buckets)
  3. call the provider with a per-attempt timeout bounded by the request deadline
  4. classify the outcome and either return it or back off and retry
Waiting for capacity (``limiter_wait``) and waiting to retry (``retry_wait``) are
measured separately from provider time, as the assignment requires.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from app.config import SchedulerSettings
from app.providers.base import (
    RATE_DAILY,
    RATE_MINUTE,
    Provider,
    ProviderResult,
)
from app.scheduler.rate_limiter import ClientLimiter, LimiterTimeout
from app.textutil import TOKENS_PER_WORD
from app.scheduler.retry import RetryPolicy

PROMPT_OVERHEAD = 70  # measured on Groq: a 2-word prompt cost 73 prompt tokens


def estimate_request_tokens(messages: list[dict[str, str]], max_tokens: int) -> int:
    """Tokens the provider will charge at admission: prompt estimate + requested max_tokens."""
    words = sum(len(m["content"].split()) for m in messages)
    return int(PROMPT_OVERHEAD + words * TOKENS_PER_WORD) + max_tokens


@dataclass
class Attempt:
    n: int
    t_start: float                 # epoch seconds
    t_end: float
    limiter_wait_s: float
    http_status: int | None
    error_type: str | None
    retry_after_s: float | None
    tokens_in: int | None
    tokens_out: int | None
    first_token_ms: float | None
    total_ms: float
    headers: dict[str, str]
    backoff_s: float = 0.0         # sleep taken *after* this attempt before the next
    decision: str = ""


@dataclass
class GatewayResult:
    result: ProviderResult | None
    attempts: list[Attempt] = field(default_factory=list)
    retries: int = 0
    limiter_wait_s: float = 0.0
    retry_wait_s: float = 0.0
    provider_ms: float = 0.0
    terminal_reason: str | None = None   # None on success
    first_token_epoch: float | None = None

    @property
    def ok(self) -> bool:
        return self.result is not None and self.result.ok and self.terminal_reason is None


class ProviderGateway:
    """Rate-limited, retrying access to a ``Provider``."""

    def __init__(self, provider: Provider, cfg: SchedulerSettings, seed: int | None = None) -> None:
        self.provider = provider
        self.cfg = cfg
        self.limiter = ClientLimiter(cfg.rpm, cfg.tpm, cfg.daily_budget, enabled=cfg.use_client_limiter)
        self.policy = RetryPolicy(cfg, seed=seed)
        self.consecutive_failures = 0
        self._unavailable_until = 0.0
        self.total_provider_calls = 0
        self.total_retries = 0

    # -- state for the decision engine ---------------------------------------
    @property
    def unavailable(self) -> bool:
        return time.monotonic() < self._unavailable_until

    def snapshot(self) -> dict[str, float | int | bool]:
        snap = self.limiter.snapshot()
        snap["provider_unavailable"] = self.unavailable
        snap["probe_due"] = self.limiter.probe_due
        snap["seconds_until_probe"] = round(self.limiter.seconds_until_probe(), 1)
        return snap

    def _note_outcome(self, res: ProviderResult) -> None:
        bad = (not res.ok) and res.error_type not in (RATE_MINUTE, RATE_DAILY)
        self.consecutive_failures = self.consecutive_failures + 1 if bad else 0
        thr = self.cfg.circuit_threshold
        if thr and self.consecutive_failures >= thr:
            self._unavailable_until = time.monotonic() + self.cfg.circuit_cooldown_s
            self.consecutive_failures = 0

    async def call(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        deadline_mono: float,
        request_id: str,
        model: str | None = None,
    ) -> GatewayResult:
        """Run the attempt/retry loop and return everything that happened."""
        out = GatewayResult(result=None)
        est = estimate_request_tokens(messages, max_tokens)
        attempt = 0
        while True:
            attempt += 1
            if self.limiter.daily_exhausted:
                out.terminal_reason = "daily_quota_exhausted"
                return out
            if self.unavailable:
                out.terminal_reason = "provider_unavailable"
                return out
            time_left = deadline_mono - time.monotonic()
            if time_left <= 0.05:
                out.terminal_reason = "deadline_exceeded"
                return out
            try:
                lw = await self.limiter.acquire(est, max_wait_s=time_left - 1.0)
            except LimiterTimeout:
                out.terminal_reason = "deadline_exceeded_waiting_for_capacity"
                return out
            out.limiter_wait_s += lw
            if attempt == 1:
                self.policy.budget.note_first_attempt()

            t_start = time.time()
            self.total_provider_calls += 1
            res = await self.provider.generate(
                messages,
                max_tokens=max_tokens,
                timeout_s=max(0.5, min(self.cfg.attempt_timeout_s, deadline_mono - time.monotonic())),
                request_key=f"{request_id}:{attempt}",
                model=model,
            )
            t_end = time.time()
            self.limiter.sync_from_headers(res.headers)
            out.provider_ms += res.total_ms
            self._note_outcome(res)
            att = Attempt(attempt, t_start, t_end, lw, res.http_status, res.error_type, res.retry_after_s,
                          res.tokens_in, res.tokens_out, res.first_token_ms, res.total_ms, dict(res.headers))
            out.attempts.append(att)
            out.result = res

            if res.ok:
                self.limiter.on_success()
                if res.tokens_in is not None and res.tokens_out is not None:
                    self.limiter.refund_tokens(est - (res.tokens_in + res.tokens_out))
                if self.limiter.probe_due or self.limiter._exhausted:
                    self.limiter.clear_daily_exhausted()
                if res.first_token_ms is not None:
                    out.first_token_epoch = t_start + res.first_token_ms / 1000.0
                att.decision = "ok"
                return out

            # --- failure -----------------------------------------------------
            if res.http_status == 429:
                self.limiter.refund_tokens(est)  # provider did not charge a rejected request
                if res.error_type == RATE_MINUTE and self.cfg.adaptive_limiter:
                    kind = "rpm" if "requests per minute" in (res.error_message or "").lower() else "tpm"
                    self.limiter.on_rate_limited(kind)
            if res.error_type == RATE_DAILY:
                self.limiter.mark_daily_exhausted(self.cfg.daily_probe_interval_s)
                att.decision = "daily_quota_exhausted"
                out.terminal_reason = "daily_quota_exhausted"
                return out
            dec = self.policy.decide(res, attempt, deadline_mono - time.monotonic())
            att.decision = dec.reason
            if not dec.retry:
                out.terminal_reason = dec.reason
                return out
            att.backoff_s = dec.wait_s
            self.policy.budget.note_retry()
            self.total_retries += 1
            out.retries += 1
            out.retry_wait_s += dec.wait_s
            await asyncio.sleep(dec.wait_s)
