"""Calibrate the stand-in's latency model from REAL provider attempts.

Reads every successful real attempt in the given runs and fits:
  first_token_ms ~= base_latency_ms + prefill_ms_per_1k_tokens * tokens_in / 1000     (+ gaussian jitter)
  tokens_per_s    = content tokens after the first / time between first token and end
  reasoning_tokens = mean hidden-reasoning tokens
Outliers (largest 10% of residuals) are trimmed because the first request on a connection and
occasional provider hiccups would otherwise dominate a small sample.

Only latency and token behaviour are calibrated. Failure rates, load-dependent slowdown and the
shape of rate-limit enforcement are NOT measured on the real service (see ``not_calibrated``).

Usage: python -m benchmarks.calibrate quality_real [real_steady real_burst ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from app.config import ROOT_DIR
from app.monitoring.metrics import load_events
from benchmarks.scenarios import CONFIG_DIR

RUNS = Path(ROOT_DIR) / "results" / "runs"


def collect(runs: list[str]) -> list[dict]:
    rows = []
    for r in runs:
        for ev in load_events(RUNS / r / "events.jsonl"):
            if ev.get("source") != "real":
                continue
            for a in ev["attempts"]:
                if a["http_status"] == 200 and a["first_token_ms"] and a["tokens_in"] and a["tokens_out"]:
                    rows.append({**a, "reasoning": ev.get("reasoning_tokens") or 0, "purpose": a["purpose"],
                                 "model": ev.get("model")})
    return rows


def fit(rows: list[dict]) -> dict:
    x = np.array([r["tokens_in"] for r in rows], float)
    y = np.array([r["first_token_ms"] for r in rows], float)
    slope, icpt = np.polyfit(x, y, 1)
    for _ in range(2):  # trim the worst 10% and refit
        res = y - (slope * x + icpt)
        keep = np.abs(res) <= np.percentile(np.abs(res), 90)
        slope, icpt = np.polyfit(x[keep], y[keep], 1)
    res = (y - (slope * x + icpt))[keep]
    slope = max(0.0, float(slope))
    icpt = float(np.mean(y[keep] - slope * x[keep]))
    gen = []
    for r in rows:
        dt = (r["total_ms"] - r["first_token_ms"]) / 1000
        content_after_first = r["tokens_out"] - r["reasoning"] - 1
        if dt > 0.02 and content_after_first > 5:
            gen.append(content_after_first / dt)
    return {
        "n_attempts": len(rows), "n_trimmed_fit": int(keep.sum()),
        "base_latency_ms": round(icpt, 1),
        "prefill_ms_per_1k_tokens": round(slope * 1000, 1),
        "latency_jitter_ms": round(float(np.std(res)), 1),
        "tokens_per_s": round(float(np.median(gen)), 1) if gen else None,
        "reasoning_tokens": round(float(np.mean([r["reasoning"] for r in rows])), 1),
        "ttft_ms_p50_p95_raw": [round(float(np.percentile(y, 50)), 0), round(float(np.percentile(y, 95)), 0)],
        "tokens_in_range": [int(x.min()), int(x.max())],
    }


def main(runs: list[str]) -> None:
    rows = collect(runs)
    if len(rows) < 8:
        raise SystemExit(f"only {len(rows)} real attempts; need >= 8 to calibrate")
    f = fit(rows)
    params = {k: f[k] for k in ("base_latency_ms", "prefill_ms_per_1k_tokens", "latency_jitter_ms",
                                "tokens_per_s", "reasoning_tokens") if f.get(k) is not None}
    out = {
        "calibrated_from": runs, "model": rows[0]["model"],
        "parameters": params, "fit_details": f,
        "taken_from_docs_or_headers": {
            "max_rpm": "30 (docs)", "max_tpm": "8,000 (docs + x-ratelimit-limit-tokens)",
            "max_daily_requests": "1,000 (docs + x-ratelimit-limit-requests)",
            "bucket behaviour": "reset headers equal used/limit*window: continuously refilling buckets",
        },
        "not_calibrated": [
            "failure_rate (no real 5xx observed; stand-in default 2% is an assumption)",
            "overload_factor / pressure model (latency growth under load was NOT measured on the real service)",
            "max_concurrency (not documented, not observed)",
            "partial_rate / hang_rate / slow_rate (no real occurrences observed)",
            "daily exhaustion behaviour (never exhausted the real daily limit)",
        ],
    }
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    p = CONFIG_DIR / "mock_calibrated.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out["parameters"], indent=1))
    print("fit:", {k: v for k, v in f.items() if k not in params})
    print("wrote", p)


if __name__ == "__main__":
    main(sys.argv[1:] or ["quality_real"])
