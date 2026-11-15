"""Shared request/job types."""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field


@dataclass
class Job:
    """One user request travelling through admission, queue and pipeline."""

    question: str
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    scenario: str = "adhoc"
    source: str = "real"                # "real" | "stand-in": which provider answers
    retrieval_setting: str = "B"
    priority: int | None = None         # lower runs first; None -> derived from question length
    deadline_s: float | None = None
    question_id: str | None = None
    # set at admission
    admit_mono: float = 0.0
    admit_epoch: float = 0.0
    future: "asyncio.Future[dict] | None" = None

    def __post_init__(self) -> None:
        if self.priority is None:
            # short questions are cheap; serving them first lowers median waiting time
            self.priority = 0 if len(self.question.split()) <= 12 else 1

    def stamp_admission(self) -> None:
        self.admit_mono = time.monotonic()
        self.admit_epoch = time.time()
