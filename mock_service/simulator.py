"""Latency model and extractive 'answer generator' for the stand-in.

The stand-in is NOT a language model. It builds answers by selecting the
best-matching sentences from the sources in the prompt and tagging them with
``[S#]``. That is enough to exercise retrieval, verification, streaming and the
scheduler under load; answer *quality* must only be judged on real Groq runs.
"""
from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass

from mock_service.limits import MockConfig

REFUSAL = "The provided research library does not contain sufficient information to answer this question."

_STOP = frozenset(
    "a an the of in on for to and or is are was were be been by with as at from that this it its "
    "which what how does do did why when where who their they we our can used using use paper "
    "papers model models between about into than then also not".split()
)
_TOK = re.compile(r"[a-z0-9]+")
_SRC = re.compile(r"^\[S(\d+)\] \(([^\n]*)\)\n(.*?)(?=\n\n\[S\d+\] \(|\n\nQuestion:|\Z)", re.S | re.M)


def toks(text: str) -> set[str]:
    return {t for t in _TOK.findall(text.lower()) if t not in _STOP and len(t) > 1}


def estimate_tokens(text: str, cfg: MockConfig) -> int:
    return round(len(text.split()) * cfg.tokens_per_word)


def prompt_tokens(messages: list[dict], cfg: MockConfig) -> int:
    return cfg.prompt_overhead_tokens + sum(estimate_tokens(str(m.get("content", "")), cfg) for m in messages)


def request_rng(cfg: MockConfig, key: str | None, fallback_n: int) -> random.Random:
    """Per-request RNG seeded from ``seed:key`` so results do not depend on arrival order."""
    k = key if key else f"anon-{fallback_n}"
    h = hashlib.sha256(f"{cfg.seed}:{k}".encode()).digest()
    return random.Random(int.from_bytes(h[:8], "big"))


@dataclass
class Plan:
    """Pre-drawn random outcomes for one request."""

    fail_500: bool
    hang: bool
    partial: bool
    partial_fraction: float
    slow: bool
    jitter_ms: float


def draw_plan(cfg: MockConfig, rng: random.Random) -> Plan:
    return Plan(
        fail_500=rng.random() < cfg.failure_rate,
        hang=rng.random() < cfg.hang_rate,
        partial=rng.random() < cfg.partial_rate,
        partial_fraction=rng.uniform(0.2, 0.8),
        slow=rng.random() < cfg.slow_rate,
        jitter_ms=rng.gauss(0.0, cfg.latency_jitter_ms),
    )


def load_multiplier(cfg: MockConfig, in_flight: int, pressure: float, slow: bool) -> float:
    """Latency multiplier: grows with concurrency and with arrivals above capacity."""
    m = 1.0 + (cfg.overload_factor - 1.0) * min(1.0, in_flight / max(1, cfg.max_concurrency))
    m += cfg.overload_factor * cfg.pressure_weight * max(0.0, pressure - 1.0)
    if slow:
        m *= cfg.slow_factor
    return m


def ttft_seconds(cfg: MockConfig, plan: Plan, ptoks: int, mult: float) -> float:
    ms = cfg.base_latency_ms + plan.jitter_ms + cfg.prefill_ms_per_1k_tokens * ptoks / 1000.0
    return max(0.05, ms * mult / 1000.0)


def parse_sources(messages: list[dict]) -> tuple[dict[int, str], str]:
    """Extract ``{n: text}`` sources and the question from the last user message."""
    user = next((m["content"] for m in reversed(messages) if m.get("role") == "user" and "[S1]" in m["content"]), "")
    srcs = {int(n): body.strip() for n, _hdr, body in _SRC.findall(user)}
    q = user.rsplit("Question:", 1)[-1].strip() if "Question:" in user else ""
    return srcs, q


def build_answer(messages: list[dict]) -> str:
    """Extractive answer with [S#] tags, or the refusal sentence if evidence is weak."""
    srcs, question = parse_sources(messages)
    qt = toks(question)
    if not srcs or not qt:
        return REFUSAL
    scored: list[tuple[float, int, str]] = []
    for n, text in srcs.items():
        for sent in re.split(r"(?<=[.!?])\s+(?=[A-Z(])", text):
            st = toks(sent)
            if len(st) < 4:
                continue
            ov = len(qt & st)
            scored.append((ov / (len(qt) ** 0.5), n, sent.strip()))
    scored.sort(key=lambda x: x[0], reverse=True)
    if not scored or len(qt & toks(scored[0][2])) < max(2, int(0.4 * len(qt))):
        return REFUSAL
    picked, used = [], set()
    for _s, n, sent in scored:
        if sent in used:
            continue
        picked.append(f"{sent} [S{n}]")
        used.add(sent)
        if len(picked) == 2:
            break
    return " ".join(picked)
