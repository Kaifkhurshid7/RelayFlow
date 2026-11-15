"""Stand-in model provider (FastAPI).

Groq-compatible ``POST /openai/v1/chat/completions`` with streaming, plus
``/admin/*`` endpoints used by the benchmark harness to change limits live,
reset state and read counters.

Run:  uvicorn mock_service.main:app --port 8001
"""
from __future__ import annotations

import asyncio
import json
import math
import time
import uuid
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from mock_service.limits import LimitState, MockConfig, fmt_duration
from mock_service.simulator import (
    build_answer,
    draw_plan,
    estimate_tokens,
    load_multiplier,
    prompt_tokens,
    request_rng,
    ttft_seconds,
)

MODEL_ECHO = "mock/stand-in"


def create_app(cfg: MockConfig | None = None) -> FastAPI:
    """Build the app; ``cfg`` defaults to ``MOCK_*`` environment variables."""
    app = FastAPI(title="RelayFlow stand-in provider")
    app.state.limits = LimitState(cfg or MockConfig())

    def err(status: int, message: str, typ: str, retry_after: float | None = None) -> JSONResponse:
        st: LimitState = app.state.limits
        headers = st.rate_headers()
        if retry_after is not None:
            headers["retry-after"] = str(max(1, math.ceil(min(86_400.0, retry_after))))
        return JSONResponse(
            {"error": {"message": message, "type": typ, "code": "rate_limit_exceeded" if status == 429 else typ}},
            status_code=status,
            headers=headers,
        )

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "cfg": app.state.limits.cfg.model_dump()}

    @app.post("/admin/config")
    async def set_config(update: dict) -> dict:
        st: LimitState = app.state.limits
        st.cfg = st.cfg.model_copy(update=update)
        st.rebuild()  # limits may have changed
        return st.cfg.model_dump()

    @app.post("/admin/reset")
    async def reset(update: dict | None = None) -> dict:
        st: LimitState = app.state.limits
        if update:
            st.cfg = st.cfg.model_copy(update=update)
        st.reset()
        return {"status": "reset", "cfg": st.cfg.model_dump()}

    @app.get("/admin/stats")
    async def stats() -> dict:
        st: LimitState = app.state.limits
        return {**st.counters.__dict__, "in_flight": st.in_flight, "pressure": st.arrival_pressure()}

    @app.post("/openai/v1/chat/completions")
    async def chat(request: Request):
        st: LimitState = app.state.limits
        cfg, c = st.cfg, st.counters
        body = await request.json()
        key = request.headers.get("x-relayflow-key")
        messages = body.get("messages", [])
        max_tokens = int(body.get("max_tokens") or 1024)
        model = body.get("model", MODEL_ECHO)
        ptoks = prompt_tokens(messages, cfg)
        need = ptoks + max_tokens

        c.arrivals += 1
        if cfg.rejected_cost_load:
            st.note_arrival()

        if need > cfg.max_tpm:
            c.rejected_too_large += 1
            return err(
                413,
                f"Request too large for model `{model}` on tokens per minute (TPM): Limit {cfg.max_tpm}, "
                f"Requested {need}, please reduce your message size and try again.",
                "tokens",
            )
        if st.in_flight >= cfg.max_concurrency:
            if cfg.concurrency_mode == "reject":
                c.rejected_concurrency += 1
                return err(429, "Too many concurrent requests for this organization.", "concurrency", 1.0)
            while st.in_flight >= cfg.max_concurrency:  # "queue" mode
                await asyncio.sleep(0.01)

        hit = st.check(need)
        if hit:
            which, wait = hit
            wait_s = fmt_duration(wait)
            if which == "rpd":
                c.rejected_daily += 1
                msg = (f"Rate limit reached for model `{model}` in organization `org_mock` service tier "
                       f"`on_demand` on requests per day (RPD): Limit {cfg.max_daily_requests}, Used "
                       f"{cfg.max_daily_requests}, Requested 1. Please try again in {wait_s}.")
                return err(429, msg, "requests", wait)
            if which == "rpm":
                c.rejected_rpm += 1
                msg = (f"Rate limit reached for model `{model}` in organization `org_mock` service tier "
                       f"`on_demand` on requests per minute (RPM): Limit {cfg.max_rpm}, Used {cfg.max_rpm}, "
                       f"Requested 1. Please try again in {wait_s}.")
                return err(429, msg, "requests", wait)
            c.rejected_tpm += 1
            used = int(cfg.max_tpm - st.tpm.level)
            msg = (f"Rate limit reached for model `{model}` in organization `org_mock` service tier "
                   f"`on_demand` on tokens per minute (TPM): Limit {cfg.max_tpm}, Used {used}, "
                   f"Requested {need}. Please try again in {wait_s}.")
            return err(429, msg, "tokens", wait)

        # admitted
        st.rpd.take(1)
        st.rpm.take(1)
        st.tpm.take(need)
        c.accepted += 1
        if not cfg.rejected_cost_load:
            st.note_arrival()
        rng = request_rng(cfg, key, c.arrivals)
        plan = draw_plan(cfg, rng)
        st.in_flight += 1
        c.peak_in_flight = max(c.peak_in_flight, st.in_flight)
        released = False

        def release() -> None:
            nonlocal released
            if not released:
                released = True
                st.in_flight = max(0, st.in_flight - 1)

        try:
            mult = load_multiplier(cfg, st.in_flight, st.arrival_pressure(), plan.slow)
            if plan.hang:
                c.hangs += 1
                await asyncio.sleep(cfg.hang_s)
            if plan.fail_500:
                c.errors_500 += 1
                await asyncio.sleep(rng.uniform(0.05, 0.2) * mult)
                release()
                return err(500, "Internal server error", "internal_server_error")
            answer = build_answer(messages)
            out_toks = estimate_tokens(answer, cfg) + cfg.reasoning_tokens
            ttft = ttft_seconds(cfg, plan, ptoks, mult)
        except BaseException:
            release()
            raise

        rid = f"req_mock_{uuid.uuid4().hex[:20]}"
        headers = st.rate_headers() | {"x-request-id": rid}

        async def stream() -> AsyncIterator[bytes]:
            def sse(obj: dict) -> bytes:
                return b"data: " + json.dumps(obj).encode() + b"\n\n"

            base = {"id": rid, "object": "chat.completion.chunk", "created": int(time.time()), "model": model}
            try:
                words = answer.split(" ")
                groups = [" ".join(words[i : i + 3]) + (" " if i + 3 < len(words) else "") for i in range(0, len(words), 3)]
                per_tok = (1.0 / cfg.tokens_per_s) * mult
                if cfg.streaming_enabled:
                    await asyncio.sleep(0.6 * ttft)
                    yield sse({**base, "choices": [{"index": 0, "delta": {"role": "assistant", "reasoning": "…"}, "finish_reason": None}]})
                    await asyncio.sleep(0.4 * ttft)
                else:
                    await asyncio.sleep(ttft + estimate_tokens(answer, cfg) * per_tok)
                    yield sse({**base, "choices": [{"index": 0, "delta": {"role": "assistant", "reasoning": "…"}, "finish_reason": None}]})
                cut_at = int(len(groups) * plan.partial_fraction) if plan.partial else None
                for i, g in enumerate(groups):
                    if cut_at is not None and i >= cut_at:
                        c.partial += 1
                        raise RuntimeError("simulated mid-stream connection loss")
                    if cfg.streaming_enabled:
                        await asyncio.sleep(estimate_tokens(g, cfg) * per_tok)
                    yield sse({**base, "choices": [{"index": 0, "delta": {"content": g}, "finish_reason": None}]})
                usage = {
                    "prompt_tokens": ptoks,
                    "completion_tokens": out_toks,
                    "total_tokens": ptoks + out_toks,
                    "completion_tokens_details": {"reasoning_tokens": cfg.reasoning_tokens},
                }
                yield sse({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                           "x_groq": {"id": rid, "usage": usage}})
                yield b"data: [DONE]\n\n"
                st.tpm.give(max(0, max_tokens - out_toks))  # unused reservation returns only on completion
                c.completed += 1
                c.tokens_served += ptoks + out_toks
            finally:
                release()

        return StreamingResponse(stream(), media_type="text/event-stream", headers=headers)

    return app


app = create_app()
