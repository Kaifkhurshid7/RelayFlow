"""Provider factory: real Groq or the local stand-in, same client code."""
from __future__ import annotations

from app.config import ProviderMode, Settings, get_settings
from app.providers.base import Provider
from app.providers.groq_client import GroqProvider
from app.providers.mock_provider import MockProvider


def make_provider(settings: Settings | None = None, mode: ProviderMode | None = None) -> Provider:
    """Build the provider for ``mode`` (defaults to ``settings.provider_mode``)."""
    s = settings or get_settings()
    m = mode or s.provider_mode
    if m is ProviderMode.MOCK:
        return MockProvider(s.mock_base_url, s.groq_model, reasoning_effort=s.reasoning_effort)
    return GroqProvider(
        s.require_groq_key(),
        s.groq_model,
        base_url=s.groq_base_url,
        reasoning_effort=s.reasoning_effort,
    )
