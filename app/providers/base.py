"""Provider-neutral result types and error classification."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

# Error classes. Retry decisions are made on these, not on raw status codes.
RATE_MINUTE = "rate_limit_minute"      # RPM / TPM: transient, honour Retry-After
RATE_DAILY = "rate_limit_daily"        # RPD / TPD: do NOT retry; budget is gone
CONCURRENCY = "rate_limit_concurrency"
SERVER = "server_error"                # 5xx
TIMEOUT = "timeout"
CONNECTION = "connection_error"
PARTIAL = "partial_response"           # stream cut after some tokens arrived
AUTH = "auth_error"
BAD_REQUEST = "bad_request"
UNKNOWN = "unknown_error"

RETRYABLE = frozenset({RATE_MINUTE, CONCURRENCY, SERVER, TIMEOUT, CONNECTION, PARTIAL})


def classify_429(message: str) -> str:
    """Tell a per-minute limit from a per-day one using Groq's error text.

    Groq's 429 body names the limit that was hit, e.g. "... on tokens per minute
    (TPM) ..." or "... on requests per day (RPD) ...".
    """
    m = message.lower()
    if "per day" in m or "(rpd)" in m or "(tpd)" in m:
        return RATE_DAILY
    if "concurren" in m:
        return CONCURRENCY
    return RATE_MINUTE


def classify_status(status: int, message: str = "") -> str:
    """Map an HTTP status (and message) to an error class."""
    if status == 429:
        return classify_429(message)
    if status in (401, 403):
        return AUTH
    if status in (400, 404, 413, 422):
        return BAD_REQUEST
    if status >= 500:
        return SERVER
    return UNKNOWN


@dataclass
class ProviderResult:
    """Outcome of one provider attempt (success or failure)."""

    ok: bool
    http_status: int | None = None
    text: str = ""
    finish_reason: str | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    reasoning_tokens: int | None = None
    headers: dict[str, str] = field(default_factory=dict)
    first_reasoning_ms: float | None = None
    first_token_ms: float | None = None   # first *content* token, from request send
    total_ms: float = 0.0
    error_type: str | None = None
    error_message: str | None = None
    retry_after_s: float | None = None
    provider_request_id: str | None = None
    partial: bool = False
    model: str | None = None


class Provider(Protocol):
    """Anything the gateway can call."""

    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        timeout_s: float,
        request_key: str | None = None,
        model: str | None = None,
    ) -> ProviderResult: ...

    async def aclose(self) -> None: ...


_RETRY_IN = re.compile(r"try again in ((?:\d+(?:\.\d+)?(?:ms|h|m|s))+)", re.I)
_UNIT = re.compile(r"(\d+(?:\.\d+)?)(ms|h|m|s)", re.I)


def parse_retry_hint(message: str) -> float | None:
    """Parse 'Please try again in 3.2s' / '250ms' / '1m26.4s' style hints into seconds."""
    m = _RETRY_IN.search(message or "")
    if not m:
        return None
    scale = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}
    return sum(float(v) * scale[u.lower()] for v, u in _UNIT.findall(m.group(1)))
