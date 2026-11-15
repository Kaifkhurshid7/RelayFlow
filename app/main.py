"""FastAPI entry point for RelayFlow.

Run:  uvicorn app.main:app --port 8000
Provider is chosen by ``PROVIDER_MODE`` (mock | groq) in ``.env``.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from app.api.routes import router
from app.config import LIBRARY, ROOT_DIR, get_settings
from app.monitoring.metrics import RunRecorder
from app.runtime import Runtime

settings = get_settings()
logging.basicConfig(
    level=settings.log_level,
    format='{"ts":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}',
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    recorder = RunRecorder(Path(ROOT_DIR) / "results" / "runs" / "service", "service")
    app.state.runtime = Runtime(settings, recorder=recorder)
    await app.state.runtime.start()
    yield
    await app.state.runtime.stop()


app = FastAPI(title="RelayFlow", version="0.2.0", lifespan=lifespan)
app.include_router(router)


@app.get("/health")
async def health() -> dict[str, object]:
    """Liveness check; reports provider mode and the exact library versions."""
    return {
        "status": "ok",
        "provider_mode": settings.provider_mode.value,
        "library": [p.label for p in LIBRARY],
    }
