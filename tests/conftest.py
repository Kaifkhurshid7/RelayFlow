"""Shared fixtures: an in-process stand-in provider on a free local port."""
from __future__ import annotations

import socket
import threading
import time

import httpx
import pytest
import uvicorn

from app.config import Settings
from mock_service.limits import MockConfig
from mock_service.main import create_app


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def mock_server():
    port = _free_port()
    app = create_app(MockConfig(base_latency_ms=60, latency_jitter_ms=0, failure_rate=0.0,
                                max_rpm=600, max_tpm=1_000_000, max_daily_requests=10_000))
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    base = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            if httpx.get(base + "/health", timeout=0.5).status_code == 200:
                break
        except httpx.HTTPError:
            time.sleep(0.05)
    yield base
    server.should_exit = True


@pytest.fixture
def mock(mock_server):
    """Reset the stand-in to a fast, fault-free baseline before each test; returns a configurer."""
    base_cfg = dict(base_latency_ms=60, latency_jitter_ms=0, failure_rate=0.0, slow_rate=0.0, hang_rate=0.0,
                    partial_rate=0.0, max_rpm=600, max_tpm=1_000_000, max_daily_requests=10_000,
                    daily_refill_per_s=0.0, max_concurrency=50, concurrency_mode="reject",
                    streaming_enabled=True, overload_factor=1.0, rejected_cost_load=True)

    def configure(**kw) -> str:
        httpx.post(mock_server + "/admin/reset", json={**base_cfg, **kw}, timeout=5)
        return mock_server

    configure()
    configure.stats = lambda: httpx.get(mock_server + "/admin/stats", timeout=5).json()  # type: ignore[attr-defined]
    return configure


@pytest.fixture
def settings(mock_server) -> Settings:
    s = Settings(_env_file=None, provider_mode="mock", mock_base_url=mock_server, analyzer_use_llm=False)
    s.scheduler.use_client_limiter = False
    s.scheduler.backoff_base_s = 0.05
    s.scheduler.backoff_max_s = 0.3
    return s
