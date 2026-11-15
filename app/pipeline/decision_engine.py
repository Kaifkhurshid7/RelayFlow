"""Decision engine: choose the execution path and its size from evidence and capacity.

Paths
  normal             full retrieval depth, normal output budget, verification on
  constrained        fewer passages / smaller output budget (optionally a cheaper
                     model) to fit available tokens, queue pressure or deadline;
                     verification stays on
  exhausted          daily quota gone: no API call, transparent message
  unavailable        provider failing fast (circuit open): no API call
  refuse_no_evidence retrieved passages do not cover the question: no API call
  clarify            question too vague to retrieve for: no API call

Principle: never trade factual correctness or citation integrity for speed. The
only things that shrink are retrieval depth and output length, and a shorter
answer must still pass verification.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from app.config import RetrievalConfig, Settings
from app.pipeline.question_analyzer import Analysis
from app.textutil import TOKENS_PER_WORD

# Weighted share of question terms that must appear in the retrieved text. Measured on the 31 evaluation
# questions: answerable >= 0.71, vocabulary-gap unanswerable <= 0.50; 0.6 sits in that gap.
# Tuned on the same set that is reported (no held-out set) - see report limitations.
COVERAGE_GATE = 0.6
MIN_K = 2
CONSTRAINED_MAX_TOKENS = 300


@dataclass
class Plan:
    path: str
    top_k: int
    max_tokens: int
    model: str | None
    reason: str
    est_tokens: int = 0


def est_prompt_tokens(passages_tokens: list[int], question: str, overhead: int = 220) -> int:
    """Estimated prompt size: system prompt + question + the passages kept."""
    return overhead + int(len(question.split()) * TOKENS_PER_WORD) + sum(passages_tokens)


def decide(
    analysis: Analysis,
    cfg: RetrievalConfig,
    passage_tokens: list[int],       # est tokens of each retrieved passage, in rank order
    coverage: float,
    state: dict,                     # ProviderGateway.snapshot()
    queue_len: int,
    queue_cap: int,
    settings: Settings,
    time_left_s: float,
    ewma_provider_s: float,
    coverage_gate: float = COVERAGE_GATE,
) -> Plan:
    """Return the plan. ``passage_tokens`` is the full retrieval (cfg.top_k scaled)."""
    full_k = len(passage_tokens)
    max_tok = settings.max_answer_tokens
    if analysis.needs_clarification:
        return Plan("clarify", 0, 0, None, analysis.clarification_reason or "vague question")
    if full_k == 0:
        return Plan("refuse_no_evidence", 0, 0, None, "no passages retrieved")
    if coverage < coverage_gate:
        return Plan("refuse_no_evidence", 0, 0, None, f"evidence coverage {coverage:.2f} < {coverage_gate:.2f}")
    if state.get("daily_exhausted"):
        return Plan("exhausted", 0, 0, None, "daily provider quota exhausted")
    if state.get("provider_unavailable"):
        return Plan("unavailable", 0, 0, None, "provider failing; fail-fast window")

    min_k = max(MIN_K, len(analysis.papers) if analysis.cross_paper else MIN_K)
    min_k = min(min_k, full_k)

    def est(k: int, mt: int) -> int:
        return est_prompt_tokens(passage_tokens[:k], analysis.question) + mt

    reasons: list[str] = []
    k, mt = full_k, max_tok
    tpm_avail = state.get("tpm_available", 10**9)
    tpm_rate = max(1.0, settings.scheduler.tpm / 60.0)
    if est(k, mt) > tpm_avail:
        # shrink passages until the request fits the tokens available right now (or hit the floor)
        while k > min_k and est(k, mt) > tpm_avail:
            k -= 1
        if est(k, mt) > tpm_avail:
            mt = CONSTRAINED_MAX_TOKENS
        reasons.append(f"tpm_available={tpm_avail} < full request {est(full_k, max_tok)}")
    if queue_cap and queue_len >= 0.6 * queue_cap:
        k = max(min_k, math.ceil(k / 2))
        mt = min(mt, CONSTRAINED_MAX_TOKENS)
        reasons.append(f"queue {queue_len}/{queue_cap}")
    if state.get("daily_remaining", 10**9) < 0.1 * settings.scheduler.daily_budget:
        k = max(min_k, math.ceil(k / 2))
        mt = min(mt, CONSTRAINED_MAX_TOKENS)
        reasons.append(f"daily_remaining={state.get('daily_remaining')} below 10% of budget")
    wait_s = max(0.0, (est(k, mt) - tpm_avail) / tpm_rate)
    if wait_s + ewma_provider_s > 0.8 * time_left_s and k > min_k:
        k = min_k
        reasons.append(f"estimated {wait_s + ewma_provider_s:.0f}s vs {time_left_s:.0f}s left")

    if reasons or k < full_k or mt < max_tok:
        return Plan("constrained", k, mt, settings.groq_model_fallback, "; ".join(reasons), est(k, mt))
    return Plan("normal", k, mt, None, "capacity available", est(k, mt))
