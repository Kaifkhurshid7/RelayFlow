"""Wires provider, gateway, pipeline, scheduler and recorder together.

One ``Runtime`` = one experiment/service instance. Benchmarks create one per run
(with its own limiter state and recorder); the API creates one at startup.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from app.config import RETRIEVAL_CONFIGS, ProviderMode, Settings, get_settings
from app.models import Job
from app.monitoring.metrics import RunRecorder
from app.pipeline.pipeline import Pipeline
from app.providers import make_provider
from app.providers.base import Provider
from app.retrieval.retrieval import Retriever
from app.scheduler.gateway import ProviderGateway
from app.scheduler.scheduler import Scheduler


@lru_cache(maxsize=1)
def get_retrievers() -> dict[str, Retriever]:
    """Build the A/B/C retrievers once per process (indexes are cheap; PDF parsing is not)."""
    return {name: Retriever.from_config(name) for name in RETRIEVAL_CONFIGS}


class Runtime:
    """A runnable RelayFlow instance."""

    def __init__(
        self,
        settings: Settings | None = None,
        mode: ProviderMode | None = None,
        recorder: RunRecorder | None = None,
        provider: Provider | None = None,
        seed: int | None = 1234,
    ) -> None:
        self.settings = settings or get_settings()
        self.mode = mode or self.settings.provider_mode
        self.source = "real" if self.mode is ProviderMode.GROQ else "stand-in"
        self.recorder = recorder
        self.provider = provider or make_provider(self.settings, self.mode)
        self.gateway = ProviderGateway(self.provider, self.settings.scheduler, seed=seed)
        self.scheduler: Scheduler  # set below (pipeline needs queue_len, scheduler needs pipeline)
        self.pipeline = Pipeline(self.settings, get_retrievers(), self.gateway, lambda: self.scheduler.queue_len)
        self.scheduler = Scheduler(
            self.settings.scheduler,
            handler=self.pipeline.run,
            on_reject=self.pipeline.rejected_event,
            on_expire=self.pipeline.expired_event,
            fast_path=lambda: self.gateway.limiter.daily_exhausted,
            on_event=self._record,
        )

    def _record(self, ev: dict) -> None:
        if self.recorder is not None:
            self.recorder.write(ev)

    async def start(self) -> None:
        await self.scheduler.start()

    async def stop(self) -> None:
        await self.scheduler.stop()
        await self.provider.aclose()

    def make_job(self, question: str, **kw) -> Job:
        kw.setdefault("source", self.source)
        kw.setdefault("retrieval_setting", self.settings.retrieval_config)
        return Job(question=question, **kw)

    async def ask(self, question: str, **kw) -> dict:
        """Submit a question and wait for its event (answer + diagnostics)."""
        fut = await self.scheduler.submit(self.make_job(question, **kw))
        return await fut
