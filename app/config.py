"""Centralised configuration for RelayFlow.

All tunables live here and can be overridden via environment variables or a
local ``.env`` file. Secrets (the Groq key) are never given a default value and
are never rendered by ``repr``/logging.
"""
from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent


class ProviderMode(str, Enum):
    """Which LLM backend the pipeline talks to."""

    MOCK = "mock"
    GROQ = "groq"


class Paper(BaseModel, frozen=True):
    """One entry of the fixed research library (exact version required)."""

    paper_id: str
    arxiv_id: str
    version: str
    title: str
    filename: str

    @property
    def url(self) -> str:
        return f"https://arxiv.org/pdf/{self.arxiv_id}{self.version}"

    @property
    def label(self) -> str:
        return f"arXiv:{self.arxiv_id}{self.version}"


LIBRARY: tuple[Paper, ...] = (
    Paper(
        paper_id="attention",
        arxiv_id="1706.03762",
        version="v7",
        title="Attention Is All You Need",
        filename="attention_is_all_you_need_v7.pdf",
    ),
    Paper(
        paper_id="moshi",
        arxiv_id="2410.00037",
        version="v1",
        title="Moshi: a speech-text foundation model for real-time dialogue",
        filename="moshi_v1.pdf",
    ),
    Paper(
        paper_id="dynamic_chunking",
        arxiv_id="2507.07955",
        version="v2",
        title="Dynamic Chunking for End-to-End Hierarchical Sequence Modeling",
        filename="dynamic_chunking_v2.pdf",
    ),
)


class RetrievalConfig(BaseModel, frozen=True):
    """A retrieval configuration (chunk size, top-k, reranking)."""

    name: str
    chunk_tokens: int
    top_k: int
    rerank: bool = False


# Starting configurations from the assignment; adjust only after measuring.
RETRIEVAL_CONFIGS: dict[str, RetrievalConfig] = {
    "A": RetrievalConfig(name="A", chunk_tokens=300, top_k=3, rerank=False),
    "B": RetrievalConfig(name="B", chunk_tokens=600, top_k=4, rerank=False),
    "C": RetrievalConfig(name="C", chunk_tokens=600, top_k=6, rerank=True),
}


class SchedulerSettings(BaseModel):
    """Admission queue, concurrency, client-side limiter and retry policy."""

    queue_capacity: int = 50
    concurrency: int = 4
    request_deadline_s: float = 90.0  # per-request deadline measured from admission
    attempt_timeout_s: float = 30.0   # per provider attempt

    # Client-side limiter. Defaults mirror the documented/observed free-tier limits
    # for openai/gpt-oss-120b (30 RPM, 8,000 TPM, 1,000 RPD).
    use_client_limiter: bool = True
    adaptive_limiter: bool = True  # AIMD: halve the client RPM rate on an RPM 429, recover slowly
    rpm: int = 30
    tpm: int = 8_000
    daily_budget: int = 1_000

    # Retry policy.
    max_retries: int = 3
    backoff_base_s: float = 0.5
    backoff_max_s: float = 8.0
    jitter: bool = True            # full jitter on exponential backoff
    honor_retry_after: bool = True
    retry_budget_ratio: float = 0.5  # retries allowed <= ratio * first attempts (+ floor)
    retry_budget_floor: int = 3
    max_repairs: int = 1           # verification-driven regeneration loop
    daily_probe_interval_s: float = 300.0  # after daily exhaustion, allow one probe call this often
    circuit_threshold: int = 8     # consecutive provider failures before fail-fast (0 = off)
    circuit_cooldown_s: float = 5.0


class Settings(BaseSettings):
    """Top-level settings. Env vars use ``__`` for nesting (``SCHEDULER__RPM``)."""

    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    provider_mode: ProviderMode = ProviderMode.MOCK
    groq_api_key: SecretStr | None = None
    groq_api_key_2: SecretStr | None = None  # loaded but intentionally unused (see README)
    groq_model: str = "openai/gpt-oss-120b"
    groq_model_fallback: str | None = None   # cheaper model for the constrained path, if configured
    groq_base_url: str | None = None
    mock_base_url: str = "http://127.0.0.1:8001"
    reasoning_effort: str = "low"
    max_answer_tokens: int = 500   # headroom: reasoning tokens share this budget
    analyzer_use_llm: bool = True
    retrieval_config: str = "B"

    scheduler: SchedulerSettings = Field(default_factory=SchedulerSettings)

    log_level: str = "INFO"

    papers_dir: Path = ROOT_DIR / "data" / "papers"
    extracted_dir: Path = ROOT_DIR / "data" / "extracted"
    indexes_dir: Path = ROOT_DIR / "data" / "indexes"
    results_dir: Path = ROOT_DIR / "results"

    def require_groq_key(self) -> str:
        """Return the Groq key or raise a clear error (never logs the value)."""
        if self.groq_api_key is None or not self.groq_api_key.get_secret_value():
            raise RuntimeError(
                "GROQ_API_KEY is not set. Put it in .env or the environment."
            )
        return self.groq_api_key.get_secret_value()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached process-wide settings."""
    return Settings()
