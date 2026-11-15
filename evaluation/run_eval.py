"""Run the quality evaluation: every selected question under each retrieval configuration.

Questions are interleaved (Q1 under A, B, C, then Q2 ...) so that if the run is stopped, every
configuration has seen the same questions. Requests are submitted one at a time, and the runner waits
OUTSIDE the system until the client TPM bucket can cover the request, so the timings recorded for each
request measure the system, not the runner's own pacing.

A hard token cap protects the free-tier daily limit (documented: 200K tokens/day).

Usage:
  python -m evaluation.run_eval --mode groq --name quality_real --ids Q01,Q03,... --configs A,B,C --max-tokens 140000
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from app.config import ROOT_DIR, ProviderMode, get_settings
from app.monitoring.metrics import RunRecorder, load_events
from app.runtime import Runtime
from evaluation.evaluator import grade_events, load_questions

NEED = {"A": 2300, "B": 3700, "C": 5200}   # estimated tokens a request of each config reserves


async def run(mode: ProviderMode, name: str, ids: list[str], configs: list[str], max_tokens: int) -> Path:
    s = get_settings()
    qs = {q["id"]: q for q in load_questions()}
    run_dir = Path(ROOT_DIR) / "results" / "runs" / name
    rec = RunRecorder(run_dir, name)
    rt = Runtime(s, mode, recorder=rec)
    await rt.start()
    spent = 0
    try:
        for qid in ids:
            for cfg in configs:
                if spent >= max_tokens:
                    print(f"STOP: token cap {max_tokens} reached after {len(rec.events)} requests")
                    raise StopIteration
                lim = rt.gateway.limiter
                wait = lim.tpm.wait_for(NEED[cfg]) if lim.enabled else 0.0
                if wait > 0:
                    await asyncio.sleep(wait + 0.2)
                ev = await rt.ask(qs[qid]["question"], retrieval_setting=cfg, scenario=f"retrieval_{cfg}",
                                  question_id=qid, deadline_s=120)
                spent += (ev["tokens_in"] or 0) + (ev["tokens_out"] or 0)
                print(f"{qid} {cfg} {ev['status']:<12} {ev['path']:<18} {ev['answer_type']:<20} "
                      f"ttft={ev['first_token_ms'] and round(ev['first_token_ms'])} total={round(ev['total_ms'])} "
                      f"tok={ev['tokens_in']}/{ev['tokens_out']} retries={ev['retries']} cum_tokens={spent}", flush=True)
    except StopIteration:
        pass
    finally:
        await rt.stop()
    # grade EVERYTHING in the run directory, so a resumed run keeps earlier events
    grades = grade_events(load_events(run_dir / "events.jsonl"))
    (run_dir / "grades.json").write_text(json.dumps(grades, indent=1), encoding="utf-8")
    print(f"done: {len(rec.events)} new requests, {spent} tokens, grades -> {run_dir / 'grades.json'}")
    return run_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["groq", "mock"], default="groq")
    ap.add_argument("--name", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--configs", default="A,B,C")
    ap.add_argument("--max-tokens", type=int, default=140_000)
    ns = ap.parse_args()
    asyncio.run(run(ProviderMode(ns.mode), ns.name, ns.ids.split(","), ns.configs.split(","), ns.max_tokens))
