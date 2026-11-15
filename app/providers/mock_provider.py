"""Client for the local stand-in service (mock_service).

It is a ``GroqProvider`` pointed at the stand-in's base URL. It also sends an
``x-relayflow-key`` header so the stand-in can derive per-request random draws
(latency, failures) from ``request_id:attempt`` and stay reproducible regardless
of arrival order under concurrency.
"""
from __future__ import annotations

from app.providers.groq_client import GroqProvider


class MockProvider(GroqProvider):
    """Groq-API client bound to the stand-in provider."""

    def __init__(self, base_url: str, model: str, *, reasoning_effort: str | None = "low") -> None:
        super().__init__(
            api_key="mock-key",
            model=model,
            base_url=base_url,
            reasoning_effort=reasoning_effort,
            send_request_key=True,
        )
