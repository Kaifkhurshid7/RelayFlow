"""Open-loop load generator.

Requests are submitted at pre-computed arrival times regardless of how fast the system answers (open
loop), which is what exposes queueing and overload; a closed-loop generator would slow itself down
and hide them. Every request ends as exactly one event in ``results/runs/<scenario>/events.jsonl``.

Usage:
  python -m benchmarks.load_generator steady burst overload rpm_halved daily_exhausted --mode mock
  python -m benchmarks.load_generator real_steady real_burst --mode groq      # uses real Groq (modest!)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx

from app.config import ROOT_DIR, ProviderMode, SchedulerSettings, Settings
from app.monitoring.metrics import RunRecorder
from app.runtime import Runtime
from benchmarks import scenarios as S
from evaluation.evaluator import grade_events

RUNS = Path(ROOT_DIR) / "results" / "runs"
CALIBRATED = S.CONFIG_DIR / "mock_calibrated.json"


def mock_config_for(sc: dict) -> dict:
    cal = json.loads(CALIBRATED.read_text(encoding="utf-8"))["parameters"] if CALIBRATED.exists() else {}
    return {**sc["mock_base"], **cal, **sc["mock"]}


async def run_scenario(key: str, mode: ProviderMode, mock_url: str = "http://127.0.0.1:8001") -> Path:
    S.write_configs()
    sc = S.load(key)
    if (sc["source"] == "real") != (mode is ProviderMode.GROQ):
        raise SystemExit(f"scenario {key} has source={sc['source']} but --mode={mode.value}")
    run_dir = RUNS / key
    if run_dir.exists():
        for f in run_dir.glob("*"):
            f.unlink()
    rec = RunRecorder(run_dir, key)

    mock_cfg: dict = {}
    if mode is ProviderMode.MOCK:
        mock_cfg = mock_config_for(sc)
        httpx.post(mock_url + "/admin/reset", json=mock_cfg, timeout=10).raise_for_status()
        settings = Settings(_env_file=None, provider_mode="mock", mock_base_url=mock_url,
                            analyzer_use_llm=False, retrieval_config=sc["retrieval"])
    else:
        settings = Settings(provider_mode="groq", retrieval_config=sc["retrieval"], analyzer_use_llm=False)
    settings.scheduler = SchedulerSettings(**{**sc["scheduler_base"], **sc["scheduler"]})

    rt = Runtime(settings, mode, recorder=rec, seed=sc["seed"])
    await rt.start()
    offsets = S.arrivals(sc)
    qids = S.questions_for(sc, len(offsets))
    from evaluation.evaluator import load_questions
    qtext = {q["id"]: q["question"] for q in load_questions()}

    queue_ts: list[list[float]] = []
    stop_mon = asyncio.Event()

    async def monitor() -> None:
        t0 = time.monotonic()
        while not stop_mon.is_set():
            queue_ts.append([round(time.monotonic() - t0, 2), rt.scheduler.queue_len, rt.scheduler.in_flight])
            await asyncio.sleep(0.5)

    mon = asyncio.create_task(monitor())
    futs: list[asyncio.Future] = []
    t_start_wall = time.time()
    t0 = time.monotonic()
    for off, qid in zip(offsets, qids):
        delay = t0 + off - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)
        job = rt.make_job(qtext[qid], scenario=key, question_id=qid, retrieval_setting=sc["retrieval"])
        futs.append(await rt.scheduler.submit(job))
        print(f"\r{key}: submitted {len(futs)}/{len(offsets)}", end="", flush=True)
    arrival_window = time.monotonic() - t0
    deadline = settings.scheduler.request_deadline_s + 60
    done, pending = await asyncio.wait(futs, timeout=deadline)
    stop_mon.set()
    await mon
    t_end_wall = time.time()
    print(f"\r{key}: {len(done)}/{len(futs)} complete" + (f", {len(pending)} UNFINISHED" if pending else ""))

    provider_stats = {}
    if mode is ProviderMode.MOCK:
        provider_stats = httpx.get(mock_url + "/admin/stats", timeout=10).json()
    meta = {
        "key": key, "label": sc["label"], "source": sc["source"], "notes": sc["notes"],
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t_start_wall)),
        "n_requests": len(futs), "n_unfinished": len(pending),
        "arrival_window_s": round(arrival_window, 2),
        "duration_s": round(t_end_wall - t_start_wall, 2),
        "offered_rate_per_min": round(len(futs) / (sc["arrival"]["duration_s"] / 60.0), 2),
        "scenario": sc, "mock_config": mock_cfg,
        "scheduler_config": settings.scheduler.model_dump(),
        "scheduler_stats": rt.scheduler.stats.__dict__,
        "gateway": {"provider_calls": rt.gateway.total_provider_calls, "retries": rt.gateway.total_retries},
        "provider_stats": provider_stats, "queue_timeseries": queue_ts,
        "sim_clock": sc.get("sim_clock"),
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    (run_dir / "grades.json").write_text(json.dumps(grade_events(rec.events), indent=1), encoding="utf-8")
    await rt.stop()
    return run_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("keys", nargs="+")
    ap.add_argument("--mode", choices=["mock", "groq"], default="mock")
    ap.add_argument("--mock-url", default="http://127.0.0.1:8001")
    ns = ap.parse_args()
    for k in ns.keys:
        asyncio.run(run_scenario(k, ProviderMode(ns.mode), ns.mock_url))
