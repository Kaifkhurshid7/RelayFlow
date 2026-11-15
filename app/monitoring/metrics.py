"""Event logging, results.csv rows and percentile statistics.

Every request produces exactly one event dict (even when rejected, timed out or
refused). Events are appended to a per-run JSONL file as they complete; the
assignment's ``results.csv`` is derived from those events, never hand-edited.
"""
from __future__ import annotations

import csv
import json
import math
import threading
from pathlib import Path
from typing import Iterable

import numpy as np

CSV_COLUMNS = [
    "request_id", "scenario", "source", "start_time", "first_token_ms", "total_ms",
    "status", "retries", "tokens_in", "tokens_out", "retrieval_setting", "answer_grade",
]
STATUSES = {"ok", "rate_limited", "timeout", "error"}
SOURCES = {"real", "stand-in"}
GRADES = {"correct", "wrong", "refused_correctly", "refused_wrongly", ""}


class RunRecorder:
    """Append-only JSONL writer for one experiment run."""

    def __init__(self, run_dir: Path, run_id: str) -> None:
        self.run_id = run_id
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "events.jsonl"
        self._lock = threading.Lock()
        self.events: list[dict] = []

    def write(self, event: dict) -> None:
        event = {"run_id": self.run_id, **event}
        with self._lock:
            self.events.append(event)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")


def load_events(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _fmt(v: object) -> object:
    if v is None:
        return ""
    if isinstance(v, float):
        return round(v, 1)
    return v


def csv_row(ev: dict) -> dict[str, object]:
    """Project an event onto the mandatory results.csv columns (empty = not applicable)."""
    row = {c: _fmt(ev.get(c)) for c in CSV_COLUMNS}
    assert row["status"] in STATUSES, row["status"]
    assert row["source"] in SOURCES, row["source"]
    assert row["answer_grade"] in GRADES, row["answer_grade"]
    return row


def write_results_csv(events: Iterable[dict], path: Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for ev in events:
            w.writerow(csv_row(ev))
            n += 1
    return n


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
MIN_N = {50: 1, 95: 20, 99: 100}  # smallest sample size at which each percentile is reported


def percentile(values: list[float], q: int) -> float | None:
    """Linear-interpolation percentile (numpy default). ``None`` when the sample is too
    small for that percentile to mean anything (p95 needs n>=20, p99 needs n>=100)."""
    vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if len(vals) < MIN_N[q]:
        return None
    return float(np.percentile(vals, q, method="linear"))


def fmt_pct(values: list[float], qs: tuple[int, ...] = (50, 95, 99)) -> str:
    """'p50/p95/p99' as whole milliseconds; 'n/m' marks a percentile the sample is too small for."""
    parts = []
    for q in qs:
        p = percentile(values, q)
        parts.append("n/m" if p is None else f"{p:.0f}")
    return "/".join(parts)
