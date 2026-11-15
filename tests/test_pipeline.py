"""Decision engine, scheduler, full pipeline (stand-in) and API tests."""
import asyncio
import csv
import json
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes import router
from app.config import RETRIEVAL_CONFIGS, ProviderMode, SchedulerSettings, Settings
from app.models import Job
from app.monitoring.metrics import CSV_COLUMNS, RunRecorder, percentile, write_results_csv
from app.pipeline.decision_engine import decide
from app.pipeline.pipeline import status_for
from app.pipeline.prompts import REFUSAL
from app.pipeline.question_analyzer import analyze_rules
from app.runtime import Runtime
from app.scheduler.scheduler import Scheduler

CFG_B = RETRIEVAL_CONFIGS["B"]
OPEN = {"tpm_available": 8000, "rpm_available": 30, "daily_remaining": 1000, "daily_exhausted": False,
        "provider_unavailable": False}


def plan(q="How many heads does the Transformer use?", toks=(600, 600, 600, 600), cov=1.0, state=None,
         queue=0, cap=50, s=None, left=60.0):
    s = s or Settings(_env_file=None)
    return decide(analyze_rules(q), CFG_B, list(toks), cov, {**OPEN, **(state or {})}, queue, cap, s, left, 1.0)


# ---- decision engine ----------------------------------------------------------------
def test_decision_normal_path() -> None:
    p = plan()
    assert p.path == "normal" and p.top_k == 4


def test_decision_refuses_without_evidence_and_never_calls_api() -> None:
    assert plan(cov=0.2).path == "refuse_no_evidence"
    assert plan(toks=()).path == "refuse_no_evidence"


def test_decision_clarify() -> None:
    assert plan(q="What about it?").path == "clarify"


def test_decision_exhausted_and_unavailable() -> None:
    assert plan(state={"daily_exhausted": True}).path == "exhausted"
    assert plan(state={"provider_unavailable": True}).path == "unavailable"


def test_decision_constrained_shrinks_retrieval_to_fit_tokens() -> None:
    p = plan(state={"tpm_available": 1900})
    assert p.path == "constrained" and 2 <= p.top_k < 4
    assert p.est_tokens <= 1900 or p.max_tokens < 500


def test_decision_constrained_on_queue_pressure_and_low_daily_budget() -> None:
    assert plan(queue=40).path == "constrained"
    q = plan(state={"daily_remaining": 50})
    assert q.path == "constrained" and q.top_k < 4


def test_decision_keeps_one_passage_per_paper_for_cross_paper() -> None:
    q = "Compare how Moshi and H-Net handle sequence length reduction"
    p = plan(q=q, toks=(600,) * 6, state={"tpm_available": 900})
    assert p.path == "constrained" and p.top_k >= 2


def test_status_mapping() -> None:
    assert status_for(None, None) == "ok"
    assert status_for("daily_quota_exhausted", "rate_limit_daily") == "rate_limited"
    assert status_for("queue_full", None) == "rate_limited"
    assert status_for("max_retries_reached", "rate_limit_minute") == "rate_limited"
    assert status_for("max_retries_reached", "server_error") == "error"
    assert status_for("deadline_exceeded", None) == "timeout"
    assert status_for("max_retries_reached", "timeout") == "timeout"
    assert status_for("bad_request_not_retryable", "bad_request") == "error"


def test_percentile_rules() -> None:
    xs = [float(i) for i in range(1, 101)]
    assert percentile(xs, 50) == pytest.approx(50.5)
    assert percentile(xs[:30], 99) is None and percentile(xs[:10], 95) is None
    assert percentile(xs, 99) == pytest.approx(99.01)


# ---- scheduler ------------------------------------------------------------------------
def make_sched(handler, **kw):
    cfg = SchedulerSettings(**{"queue_capacity": 3, "concurrency": 1, **kw})
    order: list[str] = []

    async def h(job: Job) -> dict:
        order.append(job.question)
        return await handler(job)

    sch = Scheduler(cfg, h, on_reject=lambda j: {"request_id": j.request_id, "status": "rate_limited",
                                                    "terminal_reason": "queue_full"},
                    on_expire=lambda j: {"request_id": j.request_id, "status": "timeout",
                                         "terminal_reason": "expired_in_queue"})
    return sch, order


async def test_bounded_queue_rejects_with_reason_not_silently() -> None:
    async def slow(job):
        await asyncio.sleep(0.3)
        return {"request_id": job.request_id, "status": "ok"}

    sch, _ = make_sched(slow)
    await sch.start()
    futs = [await sch.submit(Job(question="q0"))]
    await asyncio.sleep(0.05)                                            # let the worker pick up q0
    futs += [await sch.submit(Job(question=f"q{i}")) for i in range(1, 8)]   # 1 running + 3 queued + 4 rejected
    evs = await asyncio.gather(*futs)
    await sch.stop()
    assert sch.stats.rejected == 4 and sch.stats.peak_queue_len == 3
    assert [e["status"] for e in evs].count("rate_limited") == 4
    assert all(e["request_id"] for e in evs) and len(evs) == 8        # every request got an outcome


async def test_priority_and_deadline_expiry() -> None:
    gate = asyncio.Event()

    async def h(job):
        await gate.wait()
        return {"request_id": job.request_id, "status": "ok"}

    sch, order = make_sched(h, queue_capacity=10)
    await sch.start()
    first = await sch.submit(Job(question="blocker"))
    long_q = await sch.submit(Job(question="word " * 20, priority=1))
    short_q = await sch.submit(Job(question="short one", priority=0))
    doomed = await sch.submit(Job(question="doomed", priority=2, deadline_s=0.05))
    await asyncio.sleep(0.2)
    gate.set()
    evs = await asyncio.gather(first, long_q, short_q, doomed)
    await sch.stop()
    assert order[:3] == ["blocker", "short one", "word " * 20]       # priority 0 before 1
    assert evs[3]["terminal_reason"] == "expired_in_queue" and sch.stats.expired == 1


async def test_fast_path_bypasses_queue() -> None:
    gate = asyncio.Event()

    async def h(job):
        if job.question != "fast":
            await gate.wait()
        return {"request_id": job.request_id, "status": "ok"}

    cfg = SchedulerSettings(queue_capacity=5, concurrency=1)
    sch = Scheduler(cfg, h, lambda j: {}, lambda j: {}, fast_path=lambda: False)
    await sch.start()
    await sch.submit(Job(question="blocker"))
    sch._fast_path = lambda: True
    ev = await asyncio.wait_for(await sch.submit(Job(question="fast")), 1.0)   # not stuck behind the blocker
    gate.set()
    await sch.stop()
    assert ev["status"] == "ok" and sch.stats.fast_pathed == 1


# ---- full pipeline against the stand-in ------------------------------------------------
@pytest.fixture
async def rt(settings, mock, tmp_path):
    mock()
    r = Runtime(settings, ProviderMode.MOCK, recorder=RunRecorder(tmp_path / "run", "t"))
    await r.start()
    yield r
    await r.stop()


async def test_answerable_question_gets_verified_cited_answer(rt, mock) -> None:
    ev = await rt.ask("What is the inner monologue in Moshi?", retrieval_setting="B", scenario="t")
    assert ev["status"] == "ok" and ev["answer_type"] == "answer" and ev["path"] == "normal"
    assert ev["citations"] and all("arXiv:2410.00037v1" in c for c in ev["citations"])
    assert ev["verification"]["passed"] and ev["first_token_ms"] is not None
    assert 0 < ev["first_token_ms"] <= ev["total_ms"]
    assert ev["tokens_in"] > 0 and ev["tokens_out"] > 0 and ev["source"] == "stand-in"
    assert {t["tool"] for t in ev["tools"]} >= {"search_library", "evidence_coverage", "verify_answer"}
    assert mock.stats()["accepted"] == 1


async def test_unsupported_question_is_refused_without_calling_provider(rt, mock) -> None:
    ev = await rt.ask("What was Nvidia's quarterly revenue last year?", retrieval_setting="B")
    assert ev["answer"] == REFUSAL and ev["path"] == "refuse_no_evidence"
    assert ev["first_token_ms"] is None and ev["tokens_in"] is None
    assert mock.stats()["arrivals"] == 0                       # zero provider traffic


async def test_misleading_question_is_not_answered_from_general_knowledge(rt) -> None:
    ev = await rt.ask("What is the capital of France?", retrieval_setting="B")
    assert ev["answer_type"] in ("refusal_no_evidence", "refusal_model", "refusal_unverified")
    assert ev["answer"].startswith(REFUSAL)


async def test_vague_question_asks_for_clarification(rt) -> None:
    ev = await rt.ask("What about it?")
    assert ev["answer_type"] == "clarification" and ev["path"] == "clarify"


async def test_daily_quota_exhaustion_gives_transparent_message_and_stops_calls(settings, mock, tmp_path) -> None:
    mock(max_daily_requests=2, daily_refill_per_s=0.0)
    settings.scheduler.daily_probe_interval_s = 600
    r = Runtime(settings, ProviderMode.MOCK, recorder=RunRecorder(tmp_path / "d", "d"))
    await r.start()
    evs = [await r.ask("What is the inner monologue in Moshi?", retrieval_setting="B") for _ in range(5)]
    await r.stop()
    assert [e["status"] for e in evs[:2]] == ["ok", "ok"]
    assert [e["status"] for e in evs[2:]] == ["rate_limited"] * 3
    assert evs[2]["terminal_reason"] == "daily_quota_exhausted"
    assert mock.stats()["rejected_daily"] == 1                  # only ONE request ever hit the exhausted quota
    msg = evs[4]["answer"]
    assert "quota is exhausted" in msg and "NOT silently discarded" in msg and "submit it again" in msg
    assert "will be answered at" not in msg                      # no invented recovery time
    assert evs[4]["answer_type"] == "capacity_message" and evs[4]["first_token_ms"] is None


async def test_provider_failure_is_recorded_not_dropped(settings, mock, tmp_path) -> None:
    mock(failure_rate=1.0)
    settings.scheduler.max_retries = 1
    settings.scheduler.circuit_threshold = 0
    r = Runtime(settings, ProviderMode.MOCK, recorder=RunRecorder(tmp_path / "f", "f"))
    await r.start()
    ev = await r.ask("What is the inner monologue in Moshi?", retrieval_setting="B")
    await r.stop()
    assert ev["status"] == "error" and ev["retries"] == 1 and ev["terminal_reason"] == "max_retries_reached"
    assert len(ev["attempts"]) == 2 and ev["total_ms"] > 0


async def test_events_and_results_csv_have_exact_columns(rt, tmp_path) -> None:
    await rt.ask("What is the inner monologue in Moshi?", retrieval_setting="B", scenario="csvtest")
    await rt.ask("What was Nvidia's quarterly revenue?", retrieval_setting="B", scenario="csvtest")
    events = rt.recorder.events
    assert len(events) == 2
    for k in ("ts_queue_entry", "ts_processing_start", "ts_first_token", "ts_completion", "queue_ms",
              "retrieval_ms", "verification_ms", "tool_ms", "retry_wait_ms", "provider_ms", "attempts", "model"):
        assert k in events[0]
    p = tmp_path / "results.csv"
    write_results_csv(events, p)
    rows = list(csv.DictReader(p.open(encoding="utf-8")))
    assert list(rows[0].keys()) == CSV_COLUMNS and len(rows) == 2
    assert rows[1]["first_token_ms"] == "" and rows[1]["tokens_in"] == ""     # refusal: no token metrics, not zeros
    assert rows[0]["source"] == "stand-in" and rows[0]["status"] == "ok"
    json.loads((tmp_path / "run" / "events.jsonl").read_text().splitlines()[0])


# ---- API ---------------------------------------------------------------------------------
def test_api_ask_and_stream(settings, mock) -> None:
    from contextlib import asynccontextmanager

    mock()

    @asynccontextmanager
    async def lifespan(a: FastAPI):
        a.state.runtime = Runtime(settings, ProviderMode.MOCK)
        await a.state.runtime.start()
        yield
        await a.state.runtime.stop()

    app = FastAPI(lifespan=lifespan)
    app.include_router(router)
    with TestClient(app) as c:
        r = c.post("/ask", json={"question": "What is the inner monologue in Moshi?", "retrieval_setting": "B"})
        assert r.status_code == 200 and r.json()["status"] == "ok" and "Sources:" in r.json()["answer"]
        r = c.post("/ask", json={"question": "What was Nvidia's quarterly revenue?"})
        assert r.json()["answer"] == REFUSAL
        with c.stream("POST", "/ask/stream", json={"question": "What is the inner monologue in Moshi?"}) as s:
            body = "".join(s.iter_text())
        assert "event: accepted" in body and body.count("event: token") > 5 and "event: done" in body
        assert c.post("/ask", json={"question": ""}).status_code == 422
