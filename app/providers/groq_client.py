"""Async streaming client for Groq (and any Groq-API-compatible server).

The stand-in provider speaks the same API, so the *same* client code path is
used for real and simulated runs; only ``base_url`` differs.

SDK retries are disabled (``max_retries=0``): the project's own retry policy in
``app.scheduler`` is the single place that decides whether to retry.
"""
from __future__ import annotations

import asyncio
import logging
import time

import groq
import httpx
from groq import AsyncGroq

from app.providers.base import (
    CONNECTION,
    PARTIAL,
    TIMEOUT,
    UNKNOWN,
    ProviderResult,
    classify_status,
    parse_retry_hint,
)

log = logging.getLogger("relayflow.provider")

_KEEP_HEADERS = ("x-ratelimit", "retry-after", "x-groq-region", "x-request-id", "x-mock")


def _keep(headers: httpx.Headers) -> dict[str, str]:
    return {k: v for k, v in headers.items() if k.lower().startswith(_KEEP_HEADERS)}


def _retry_after(headers: httpx.Headers, message: str) -> float | None:
    ra = headers.get("retry-after")
    if ra:
        try:
            return float(ra)
        except ValueError:
            pass
    return parse_retry_hint(message)


class GroqProvider:
    """Streaming chat-completions client with timing, headers and usage capture."""

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        base_url: str | None = None,
        reasoning_effort: str | None = "low",
        send_request_key: bool = False,
    ) -> None:
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.send_request_key = send_request_key
        self._client = AsyncGroq(api_key=api_key, base_url=base_url, max_retries=0)

    async def aclose(self) -> None:
        await self._client.close()

    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        timeout_s: float,
        request_key: str | None = None,
        model: str | None = None,
    ) -> ProviderResult:
        """Run one streaming completion; never raises, always returns a result."""
        use_model = model or self.model
        res = ProviderResult(ok=False, model=use_model)
        t0 = time.perf_counter()
        parts: list[str] = []
        extra = {"x-relayflow-key": request_key} if (self.send_request_key and request_key) else None
        kwargs: dict[str, object] = {}
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        try:
            async with asyncio.timeout(timeout_s):
                raw = await self._client.chat.completions.with_raw_response.create(
                    model=use_model,
                    messages=messages,  # type: ignore[arg-type]
                    max_tokens=max_tokens,
                    stream=True,
                    extra_headers=extra,
                    **kwargs,  # type: ignore[arg-type]
                )
                res.http_status = raw.status_code
                res.headers = _keep(raw.headers)
                stream = await raw.parse()
                async for chunk in stream:
                    now = (time.perf_counter() - t0) * 1000
                    if chunk.choices:
                        ch = chunk.choices[0]
                        d = ch.delta
                        if getattr(d, "reasoning", None) and res.first_reasoning_ms is None:
                            res.first_reasoning_ms = now
                        if d.content:
                            if res.first_token_ms is None:
                                res.first_token_ms = now
                            parts.append(d.content)
                        if ch.finish_reason:
                            res.finish_reason = ch.finish_reason
                    usage = getattr(chunk, "usage", None) or getattr(
                        getattr(chunk, "x_groq", None), "usage", None
                    )
                    if usage is not None:
                        res.tokens_in = usage.prompt_tokens
                        res.tokens_out = usage.completion_tokens
                        det = getattr(usage, "completion_tokens_details", None)
                        res.reasoning_tokens = getattr(det, "reasoning_tokens", None)
                    xg = getattr(chunk, "x_groq", None)
                    if xg is not None and getattr(xg, "id", None):
                        res.provider_request_id = xg.id
                res.text = "".join(parts)
                res.ok = True
        except TimeoutError:
            res.text = "".join(parts)
            res.partial = bool(parts)
            res.error_type = TIMEOUT
            res.error_message = f"attempt exceeded {timeout_s:.1f}s"
        except groq.APIStatusError as e:
            msg = getattr(e, "message", str(e))
            res.http_status = e.status_code
            res.headers = _keep(e.response.headers)
            res.error_type = classify_status(e.status_code, msg)
            res.error_message = msg[:500]
            res.retry_after_s = _retry_after(e.response.headers, msg)
        except groq.APITimeoutError:
            res.error_type, res.error_message = TIMEOUT, "client read timeout"
        except (groq.APIConnectionError, httpx.HTTPError) as e:
            res.text = "".join(parts)
            res.partial = bool(parts)
            res.error_type = PARTIAL if parts else CONNECTION
            res.error_message = f"{type(e).__name__}: {e}"[:300]
        except Exception as e:  # noqa: BLE001 - provider layer must never raise
            log.exception("unexpected provider error")
            res.error_type, res.error_message = UNKNOWN, f"{type(e).__name__}: {e}"[:300]
        res.total_ms = (time.perf_counter() - t0) * 1000
        if res.ok and res.finish_reason == "length" and not res.text.strip():
            # reasoning consumed the whole token budget: not a usable answer
            res.ok = False
            res.error_type = "empty_response"
            res.error_message = "finish_reason=length with empty content (reasoning ate max_tokens)"
        return res
