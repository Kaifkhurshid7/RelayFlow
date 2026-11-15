"""The RelayFlow pipeline: one request from admission to a logged outcome.

Stages
  1. question analysis      (rules; optional LLM assist; deterministic fallback)
  2. retrieval              (tool: BM25 / balanced per-paper loop / widen-scope loop)
  3. decision               (normal | constrained | exhausted | unavailable | refuse | clarify)
  4. generation             (via ProviderGateway: limiter, retries, deadline)
  5. verification           (tool: deterministic citation + support checks)
     +- repair loop         (bounded re-generation when verification fails)
  6. response               (answer + expanded citations, streamed by the API layer)
  7. logging                (one event per request, success or not)

Tools are real code executed by the pipeline and timed individually
(``search_library``, ``evidence_coverage``, ``verify_answer``); they are invoked
by the orchestrator, not through model function-calling, to save tokens and
latency on a 8,000-TPM budget.
"""
from __future__ import annotations

import math
import time
from typing import Any

from app.config import RETRIEVAL_CONFIGS, Settings
from app.models import Job
from app.pipeline import messages as M
from app.pipeline.decision_engine import COVERAGE_GATE, Plan, decide
from app.pipeline.prompts import REFUSAL, build_messages
from app.pipeline.retrieval_step import retrieve
from app.pipeline.question_analyzer import (
    LLM_SYSTEM,
    Analysis,
    analyze_rules,
    parse_llm_analysis,
)
from app.providers.base import CONCURRENCY, RATE_DAILY, RATE_MINUTE, TIMEOUT
from app.retrieval.retrieval import Passage, Retriever
from app.scheduler.gateway import GatewayResult, ProviderGateway
from app.verification.verification import VerificationResult, normalize_model_text, verify_answer

GIVEN_UP_REASONS = {
    "max_retries_reached", "retry_budget_exhausted", "deadline_would_be_exceeded",
    "deadline_exceeded", "deadline_exceeded_waiting_for_capacity", "expired_in_queue",
}


def status_for(reason: str | None, last_error: str | None) -> str:
    """Map a terminal reason (+ last provider error class) onto results.csv ``status``."""
    if reason is None:
        return "ok"
    if reason in ("daily_quota_exhausted", "queue_full"):
        return "rate_limited"
    if reason in ("max_retries_reached", "retry_budget_exhausted") and last_error in (RATE_MINUTE, RATE_DAILY, CONCURRENCY):
        return "rate_limited"
    if reason.startswith("deadline") or reason == "expired_in_queue" or last_error == TIMEOUT:
        return "timeout"
    return "error"


class Pipeline:
    """Holds the long-lived components and runs jobs."""

    def __init__(
        self,
        settings: Settings,
        retrievers: dict[str, Retriever],
        gateway: ProviderGateway,
        queue_len_fn=lambda: 0,
    ) -> None:
        self.s = settings
        self.retrievers = retrievers
        self.gateway = gateway
        self.queue_len = queue_len_fn
        self.ewma_provider_s = 1.0

    # ------------------------------------------------------------------ events
    def _new_event(self, job: Job) -> dict[str, Any]:
        return {
            "request_id": job.request_id, "scenario": job.scenario, "source": job.source,
            "question_id": job.question_id, "question": job.question,
            "retrieval_setting": job.retrieval_setting, "priority": job.priority,
            "start_time": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(job.admit_epoch))
            + f".{int((job.admit_epoch % 1) * 1000):03d}Z",
            "ts_queue_entry": job.admit_epoch, "ts_processing_start": None,
            "ts_first_token": None, "ts_completion": None,
            "queue_ms": None, "analysis_ms": 0.0, "retrieval_ms": 0.0, "decision_ms": 0.0,
            "limiter_wait_ms": 0.0, "provider_ms": 0.0, "retry_wait_ms": 0.0,
            "verification_ms": 0.0, "tool_ms": 0.0,
            "first_token_ms": None, "total_ms": None, "client_first_byte_ms": None,
            "status": None, "terminal_reason": None, "answer_type": "none", "path": None,
            "plan": None, "analysis": None, "retries": 0, "llm_calls": 0, "repairs": 0,
            "tokens_in": None, "tokens_out": None, "reasoning_tokens": None,
            "model": None, "http_status_final": None, "error_type_final": None,
            "attempts": [], "tools": [], "verification": None, "answer": "", "citations": [], "citation_chunks": [], "generations": [],
            "answer_grade": "", "headers_last": {},
        }

    def _absorb(self, ev: dict, g: GatewayResult, purpose: str) -> None:
        """Fold one gateway call into the event totals."""
        ev["limiter_wait_ms"] += g.limiter_wait_s * 1000
        ev["retry_wait_ms"] += g.retry_wait_s * 1000
        ev["provider_ms"] += g.provider_ms
        ev["retries"] += g.retries
        for a in g.attempts:
            ev["attempts"].append({"purpose": purpose, **a.__dict__})
        if g.result is not None:
            r = g.result
            ev["http_status_final"], ev["error_type_final"] = r.http_status, r.error_type
            ev["headers_last"] = r.headers or ev["headers_last"]
            ev["model"] = r.model
            if r.ok:
                ev["llm_calls"] += 1
                for k, v in (("tokens_in", r.tokens_in), ("tokens_out", r.tokens_out),
                             ("reasoning_tokens", r.reasoning_tokens)):
                    if v is not None:
                        ev[k] = (ev[k] or 0) + v
                if purpose == "generate" and ev["ts_first_token"] is None and g.first_token_epoch:
                    ev["ts_first_token"] = g.first_token_epoch
                    ev["first_token_ms"] = (g.first_token_epoch - ev["ts_queue_entry"]) * 1000
                if g.provider_ms:
                    self.ewma_provider_s = 0.7 * self.ewma_provider_s + 0.3 * g.provider_ms / 1000

    def _finish(self, job: Job, ev: dict, t_end_mono: float, *, status: str, reason: str | None) -> dict:
        ev["status"], ev["terminal_reason"] = status, reason
        ev["total_ms"] = (t_end_mono - job.admit_mono) * 1000
        ev["ts_completion"] = job.admit_epoch + ev["total_ms"] / 1000
        ev["client_first_byte_ms"] = ev["total_ms"]  # answer is streamed after verification
        ev["tool_ms"] = sum(t["ms"] for t in ev["tools"])
        return ev

    def _tool(self, ev: dict, name: str, fn, *a, **kw):
        t0 = time.perf_counter()
        out = fn(*a, **kw)
        ms = (time.perf_counter() - t0) * 1000
        ev["tools"].append({"tool": name, "ms": round(ms, 3)})
        return out, ms

    # ------------------------------------------------------- non-run outcomes
    def rejected_event(self, job: Job, reason: str = "queue_full") -> dict:
        ev = self._new_event(job)
        ev["answer"], ev["answer_type"] = M.rejected_queue_full_message(job.request_id), "capacity_message"
        ev["path"] = "rejected_at_admission"
        return self._finish(job, ev, time.monotonic(), status="rate_limited", reason=reason)

    def expired_event(self, job: Job) -> dict:
        ev = self._new_event(job)
        now = time.monotonic()
        ev["ts_processing_start"], ev["queue_ms"] = time.time(), (now - job.admit_mono) * 1000
        ev["answer"], ev["answer_type"] = M.timeout_message(job.request_id), "capacity_message"
        ev["path"] = "expired_in_queue"
        return self._finish(job, ev, now, status="timeout", reason="expired_in_queue")

    # -------------------------------------------------------------------- run
    async def run(self, job: Job) -> dict:
        ev = self._new_event(job)
        t_start = time.monotonic()
        ev["ts_processing_start"] = time.time()
        ev["queue_ms"] = (t_start - job.admit_mono) * 1000
        deadline = job.admit_mono + (job.deadline_s or self.s.scheduler.request_deadline_s)
        cfg = RETRIEVAL_CONFIGS[job.retrieval_setting]
        retr = self.retrievers[job.retrieval_setting]
        q = job.question

        # 1. analysis ---------------------------------------------------------
        a = analyze_rules(q)
        state = self.gateway.snapshot()
        if (self.s.analyzer_use_llm and not a.papers and not a.needs_clarification
                and not state["daily_exhausted"] and state["tpm_available"] > 2000):
            a = await self._llm_assist(job, ev, a, deadline)
        ev["analysis_ms"] = a.duration_ms
        ev["analysis"] = {k: v for k, v in a.__dict__.items() if k not in ("question", "duration_ms")}

        # 2. retrieval (tools: search_library, evidence_coverage) -- shared with the offline evaluation ----
        def run_tool(name: str, fn, *args):
            out, ms = self._tool(ev, name, fn, *args)
            ev["retrieval_ms"] += ms
            return out

        passages, coverage, scope_used, k_full = retrieve(retr, a, q, run_tool)

        # 3. decision -------------------------------------------------------------
        t0 = time.perf_counter()
        plan = decide(
            a, cfg, [p.chunk.n_tokens_est for p in passages], coverage, self.gateway.snapshot(),
            self.queue_len(), self.s.scheduler.queue_capacity, self.s,
            max(0.0, deadline - time.monotonic()), self.ewma_provider_s,
        )
        ev["decision_ms"] = (time.perf_counter() - t0) * 1000
        ev["path"] = plan.path
        ev["plan"] = {**plan.__dict__, "coverage": round(coverage, 3), "scope": scope_used, "k_full": k_full,
                      "passages": [{"chunk_id": p.chunk.chunk_id, "score": round(p.score, 2),
                                    "paper": p.chunk.paper_id, "section": p.chunk.section,
                                    "pages": [p.chunk.page_start, p.chunk.page_end]} for p in passages]}

        # 4. non-LLM paths ---------------------------------------------------------
        if plan.path == "refuse_no_evidence":
            ev["answer"], ev["answer_type"] = REFUSAL, "refusal_no_evidence"
            return self._finish(job, ev, time.monotonic(), status="ok", reason=None)
        if plan.path == "clarify":
            ev["answer"], ev["answer_type"] = M.clarify_message(a.clarification_reason or ""), "clarification"
            return self._finish(job, ev, time.monotonic(), status="ok", reason=None)
        if plan.path == "exhausted":
            ev["answer"] = M.exhausted_message(job.request_id, self.gateway.limiter.seconds_until_probe())
            ev["answer_type"] = "capacity_message"
            return self._finish(job, ev, time.monotonic(), status="rate_limited", reason="daily_quota_exhausted")
        if plan.path == "unavailable":
            ev["answer"], ev["answer_type"] = M.unavailable_message(job.request_id), "capacity_message"
            return self._finish(job, ev, time.monotonic(), status="error", reason="provider_unavailable")

        # 5. generation + verification + bounded repair loop ----------------------------
        used = passages[: plan.top_k]
        msgs = build_messages(q, used)
        v: VerificationResult | None = None
        text = ""
        for attempt in range(1 + self.s.scheduler.max_repairs):
            g = await self.gateway.call(msgs, max_tokens=plan.max_tokens, deadline_mono=deadline,
                                        request_id=job.request_id, model=plan.model)
            self._absorb(ev, g, "generate" if attempt == 0 else "repair")
            if not g.ok:
                last = g.result.error_type if g.result else None
                reason = g.terminal_reason or "provider_error"
                if reason.endswith("_not_retryable") or reason == "provider_error":
                    reason = last or reason
                ev["answer"] = M.exhausted_message(job.request_id, self.gateway.limiter.seconds_until_probe()) \
                    if reason == "daily_quota_exhausted" else M.timeout_message(job.request_id) \
                    if status_for(reason, last) == "timeout" else (
                        "The model provider returned an error and no answer could be produced "
                        f"(reference {job.request_id}). Please resubmit.")
                ev["answer_type"] = "capacity_message"
                return self._finish(job, ev, time.monotonic(), status=status_for(reason, last), reason=reason)
            text = normalize_model_text(g.result.text).strip()
            v, vms = self._tool(ev, "verify_answer", verify_answer, text, q, used, set(a.named_papers))
            ev["verification_ms"] += vms
            ev["generations"].append({"attempt": attempt + 1, "text": text, "passed": v.passed, "issues": v.issues})
            ev["verification"] = {"passed": v.passed, "is_refusal": v.is_refusal, "issues": v.issues,
                                  "warnings": v.warnings, "min_support": v.min_support}
            if v.passed:
                break
            if attempt < self.s.scheduler.max_repairs and deadline - time.monotonic() > 5:
                ev["repairs"] += 1
                msgs = build_messages(q, used, v.feedback(), text)
        assert v is not None

        # 6. response -----------------------------------------------------------------
        if v.passed and v.is_refusal:
            ev["answer"], ev["answer_type"] = REFUSAL, "refusal_model"
        elif v.passed:
            ev["citations"] = [c.render() for c in v.citations]
            ev["citation_chunks"] = [{"chunk_id": c.chunk_id, "paper_id": c.paper_id,
                                      "pages": [c.page_start, c.page_end], "section": c.section}
                                     for c in v.citations]
            ev["answer"] = text + "\n\nSources:\n" + "\n".join(ev["citations"])
            ev["answer_type"] = "answer"
        else:
            ev["answer"] = REFUSAL + " (A draft answer was produced but failed citation/evidence verification, so it was withheld.)"
            ev["answer_type"] = "refusal_unverified"
        return self._finish(job, ev, time.monotonic(), status="ok", reason=None)

    # ------------------------------------------------------------- LLM assist
    async def _llm_assist(self, job: Job, ev: dict, base: Analysis, deadline: float) -> Analysis:
        """Optional model call for questions the rules cannot place; never fatal."""
        msgs = [{"role": "system", "content": LLM_SYSTEM}, {"role": "user", "content": job.question}]
        g = await self.gateway.call(msgs, max_tokens=200, deadline_mono=min(deadline, time.monotonic() + 15),
                                    request_id=job.request_id + "-an")
        self._absorb(ev, g, "analyze")
        if g.ok and g.result:
            merged = parse_llm_analysis(g.result.text, base)
            if merged is not None:
                return merged
        return base  # fallback: rules only
