"""HTTP API: ``POST /ask`` (JSON) and ``POST /ask/stream`` (server-sent events)."""
from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

router = APIRouter()


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    retrieval_setting: str | None = Field(default=None, pattern="^[ABC]$")


def _public(ev: dict) -> dict:
    """The part of an event that is safe and useful to return to a caller."""
    keys = ("request_id", "status", "answer_type", "answer", "citations", "path",
            "terminal_reason", "retries", "first_token_ms", "total_ms", "queue_ms")
    return {k: ev.get(k) for k in keys}


@router.post("/ask")
async def ask(body: AskRequest, request: Request) -> JSONResponse:
    rt = request.app.state.runtime
    kw = {"retrieval_setting": body.retrieval_setting} if body.retrieval_setting else {}
    ev = await rt.ask(body.question, **kw)
    code = {"ok": 200, "rate_limited": 429, "timeout": 504, "error": 502}[ev["status"]]
    return JSONResponse(_public(ev), status_code=code)


@router.post("/ask/stream")
async def ask_stream(body: AskRequest, request: Request) -> StreamingResponse:
    """Stream the *verified* answer.

    The model's output is verified in full before any of it reaches the caller,
    so the stream starts after verification (``client_first_byte_ms`` in the event
    log records this). Provider-side first-token time is measured separately.
    """
    rt = request.app.state.runtime
    kw = {"retrieval_setting": body.retrieval_setting} if body.retrieval_setting else {}

    async def gen() -> AsyncIterator[str]:
        yield "event: accepted\ndata: {}\n\n"
        ev = await rt.ask(body.question, **kw)
        for word in ev["answer"].split(" "):
            yield f"event: token\ndata: {json.dumps(word + ' ')}\n\n"
            await asyncio.sleep(0)
        meta = {k: v for k, v in _public(ev).items() if k != "answer"}
        yield f"event: done\ndata: {json.dumps(meta)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")
