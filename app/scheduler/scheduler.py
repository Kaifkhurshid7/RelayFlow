"""Admission control and the bounded priority queue.

Every submitted job ends in exactly one of these ways, and each is recorded:
  * accepted immediately  - a worker was idle and the queue empty
  * queued then processed - waited for a worker (queue wait measured)
  * rejected              - queue full: answered at once with a reason, never dropped silently
  * expired               - its deadline passed while it waited in the queue
  * fast-pathed           - daily quota known exhausted: answered at once with the
                            transparent message instead of waiting behind running work
"""
from __future__ import annotations

import asyncio
import heapq
import itertools
import logging
from dataclasses import dataclass
from typing import Awaitable, Callable

from app.config import SchedulerSettings
from app.models import Job

log = logging.getLogger("relayflow.scheduler")

Handler = Callable[[Job], Awaitable[dict]]


@dataclass
class SchedulerStats:
    submitted: int = 0
    accepted_immediately: int = 0
    queued: int = 0
    rejected: int = 0
    expired: int = 0
    fast_pathed: int = 0
    peak_queue_len: int = 0
    peak_in_flight: int = 0


class Scheduler:
    """Bounded priority queue + fixed worker pool."""

    def __init__(
        self,
        cfg: SchedulerSettings,
        handler: Handler,
        on_reject: Callable[[Job], dict],
        on_expire: Callable[[Job], dict],
        fast_path: Callable[[], bool] = lambda: False,
        on_event: Callable[[dict], None] = lambda ev: None,
    ) -> None:
        self.cfg = cfg
        self._handler, self._on_reject, self._on_expire = handler, on_reject, on_expire
        self._fast_path, self._on_event = fast_path, on_event
        self._heap: list[tuple[int, int, Job]] = []
        self._seq = itertools.count()
        self._cv = asyncio.Condition()
        self._workers: list[asyncio.Task] = []
        self._inline: set[asyncio.Task] = set()
        self._closed = False
        self.in_flight = 0
        self.stats = SchedulerStats()

    # -- lifecycle -------------------------------------------------------------
    async def start(self) -> None:
        self._closed = False
        self._workers = [asyncio.create_task(self._worker(i)) for i in range(self.cfg.concurrency)]

    async def stop(self) -> None:
        self._closed = True
        async with self._cv:
            self._cv.notify_all()
        for t in [*self._workers, *self._inline]:
            t.cancel()
        await asyncio.gather(*self._workers, *self._inline, return_exceptions=True)

    @property
    def queue_len(self) -> int:
        return len(self._heap)

    # -- admission -------------------------------------------------------------
    async def submit(self, job: Job) -> "asyncio.Future[dict]":
        """Admit ``job`` and return a future resolved with its event dict."""
        job.stamp_admission()
        job.future = asyncio.get_running_loop().create_future()
        st = self.stats
        st.submitted += 1

        if self._fast_path():
            st.fast_pathed += 1
            t = asyncio.create_task(self._run(job))
            self._inline.add(t)
            t.add_done_callback(self._inline.discard)
            return job.future
        if len(self._heap) >= self.cfg.queue_capacity:
            st.rejected += 1
            self._resolve(job, self._on_reject(job))
            return job.future
        if self.in_flight < self.cfg.concurrency and not self._heap:
            st.accepted_immediately += 1
        else:
            st.queued += 1
        async with self._cv:
            heapq.heappush(self._heap, (job.priority or 0, next(self._seq), job))
            st.peak_queue_len = max(st.peak_queue_len, len(self._heap))
            self._cv.notify()
        return job.future

    # -- execution -------------------------------------------------------------
    def _resolve(self, job: Job, ev: dict) -> None:
        self._on_event(ev)
        if job.future and not job.future.done():
            job.future.set_result(ev)

    async def _run(self, job: Job) -> None:
        try:
            self._resolve(job, await self._handler(job))
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 - one bad request must not kill a worker
            log.exception("handler crashed for %s", job.request_id)
            ev = self._on_reject(job)
            ev.update(status="error", terminal_reason=f"internal_error:{type(e).__name__}")
            self._resolve(job, ev)

    async def _worker(self, idx: int) -> None:
        import time

        while not self._closed:
            async with self._cv:
                while not self._heap and not self._closed:
                    await self._cv.wait()
                if self._closed:
                    return
                _p, _s, job = heapq.heappop(self._heap)
            deadline = job.admit_mono + (job.deadline_s or self.cfg.request_deadline_s)
            if time.monotonic() >= deadline:
                self.stats.expired += 1
                self._resolve(job, self._on_expire(job))
                continue
            self.in_flight += 1
            self.stats.peak_in_flight = max(self.stats.peak_in_flight, self.in_flight)
            try:
                await self._run(job)
            finally:
                self.in_flight -= 1
