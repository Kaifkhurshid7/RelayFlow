"""Scenario definitions (pure data + deterministic arrival generation).

Each scenario is a JSON-serialisable dict, written to ``experiments/configurations/`` by
``python -m benchmarks.scenarios --write`` and read back by the load generator, so the exact
parameters of every experiment are on disk next to the results.

Stand-in capacity (documented in the report):
  * RPM 30 mirrors the documented free-tier limit for openai/gpt-oss-120b.
  * TPM is raised to 60,000 (real: 8,000) so that the REQUEST limit - not the token limit - is the
    binding constraint; at the real 8,000 TPM the free tier admits only ~4 requests/min, which would
    make scenarios C-D indistinguishable. This deviation is deliberate and labelled in the report.
  * Latency parameters come from experiments/configurations/mock_calibrated.json when it exists.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from app.config import ROOT_DIR

CONFIG_DIR = Path(ROOT_DIR) / "experiments" / "configurations"

# Questions that reach the provider (answerable + the two refusals the model must decide itself).
POOL = ["Q01", "Q03", "Q05", "Q06", "Q07", "Q09", "Q10", "Q13", "Q17", "Q22", "Q28", "Q29", "Q31"]

BASE_MOCK = {
    "max_rpm": 30, "max_tpm": 60_000, "max_daily_requests": 1_000_000, "daily_refill_per_s": 0.0,
    "max_concurrency": 10, "concurrency_mode": "reject", "failure_rate": 0.02, "overload_factor": 1.5,
    "pressure_weight": 0.5, "rejected_cost_load": True, "streaming_enabled": True, "seed": 1234,
}
BASE_SCHED = {
    "queue_capacity": 40, "concurrency": 8, "request_deadline_s": 60.0, "attempt_timeout_s": 15.0,
    "use_client_limiter": True, "rpm": 30, "tpm": 60_000, "daily_budget": 1_000_000,
    "max_retries": 3, "daily_probe_interval_s": 30.0,
}

SCENARIOS: dict[str, dict] = {
    "steady": {
        "label": "Steady", "source": "stand-in", "retrieval": "A", "seed": 11,
        "arrival": {"kind": "steady", "rpm": 24, "duration_s": 150},
        "mock": {}, "scheduler": {},
        "notes": "24 requests/min (80% of the 30 RPM limit) for 150 s.",
    },
    "burst": {
        "label": "Burst", "source": "stand-in", "retrieval": "A", "seed": 12,
        "arrival": {"kind": "burst", "base_rpm": 10, "duration_s": 150, "burst_at_s": 40, "burst_n": 60,
                    "burst_window_s": 5},
        "mock": {}, "scheduler": {},
        "notes": "10 requests/min background, plus 60 requests arriving within 5 s at t=40 s.",
    },
    "overload": {
        "label": "Sustained overload", "source": "stand-in", "retrieval": "A", "seed": 13,
        "arrival": {"kind": "steady", "rpm": 60, "duration_s": 180},
        "mock": {}, "scheduler": {},
        "notes": "60 requests/min (2x the 30 RPM limit) for 180 s.",
    },
    "rpm_halved": {
        "label": "Per-minute limit halved", "source": "stand-in", "retrieval": "A", "seed": 14,
        "arrival": {"kind": "steady", "rpm": 24, "duration_s": 150},
        "mock": {"max_rpm": 15}, "scheduler": {},
        "notes": "Same offered load as 'steady', but the provider silently enforces 15 RPM; the client still "
                 "believes 30 and must discover the new limit from 429 responses.",
    },
    "daily_exhausted": {
        "label": "Daily budget exhausted", "source": "stand-in", "retrieval": "A", "seed": 15,
        "arrival": {"kind": "steady", "rpm": 20, "duration_s": 240},
        "mock": {"max_rpm": 120, "max_daily_requests": 60, "daily_refill_per_s": 0.0},
        "scheduler": {"rpm": 120},
        "sim_clock": {"start": "09:00", "end": "17:00", "duration_s": 240, "exhausted_label": "15:00"},
        "notes": "Compressed day: 240 s stands for 09:00-17:00, so 15:00 is t=180 s. The provider's daily budget "
                 "(60 requests, no refill) runs out at about t=180 s.",
    },
    # --- modest REAL-provider runs (free tier: 8,000 TPM, 200K tokens/day: kept deliberately small) ---------
    "real_steady": {
        "label": "Steady", "source": "real", "retrieval": "A", "seed": 31,
        "arrival": {"kind": "steady", "rpm": 4, "duration_s": 120},
        "mock": {},
        "scheduler": {"tpm": 8000, "rpm": 30, "daily_budget": 1000, "concurrency": 6, "queue_capacity": 20,
                      "request_deadline_s": 90.0, "attempt_timeout_s": 30.0},
        "notes": "REAL Groq, 4 requests/min for 120 s (8 requests): about what 8,000 TPM sustains with config A.",
    },
    "real_burst": {
        "label": "Burst", "source": "real", "retrieval": "A", "seed": 32,
        "arrival": {"kind": "burst", "base_rpm": 0.0001, "duration_s": 1, "burst_at_s": 0, "burst_n": 6,
                    "burst_window_s": 1},
        "mock": {},
        "scheduler": {"tpm": 8000, "rpm": 30, "daily_budget": 1000, "concurrency": 6, "queue_capacity": 20,
                      "request_deadline_s": 90.0, "attempt_timeout_s": 30.0, "use_client_limiter": False,
                      "max_retries": 4},
        "notes": "REAL Groq, 6 requests within 1 s with the client limiter OFF, so the provider's own 429 / "
                 "Retry-After behaviour is observed; retries are on.",
    },
    # --- retry experiment: same overloaded provider, three client behaviours -----------------------------
    "retry_none": {
        "label": "Overload, no retries", "source": "stand-in", "retrieval": "A", "seed": 21,
        "arrival": {"kind": "steady", "rpm": 45, "duration_s": 120},
        "mock": {}, "scheduler": {"use_client_limiter": False, "max_retries": 0, "circuit_threshold": 0},
        "notes": "Client sends everything it is given and never retries.",
    },
    "retry_naive": {
        "label": "Overload, naive retries", "source": "stand-in", "retrieval": "A", "seed": 21,
        "arrival": {"kind": "steady", "rpm": 45, "duration_s": 120},
        "mock": {},
        "scheduler": {"use_client_limiter": False, "max_retries": 5, "backoff_base_s": 0.05, "backoff_max_s": 0.05,
                      "jitter": False, "honor_retry_after": False, "retry_budget_ratio": 1000.0,
                      "retry_budget_floor": 100000, "circuit_threshold": 0, "attempt_timeout_s": 5.0,
                      "queue_capacity": 400, "concurrency": 40, "adaptive_limiter": False},
        "notes": "Immediate retries (50 ms, no jitter), ignores Retry-After, up to 5 per request, no retry budget, "
                 "no client limiter.",
    },
    "retry_relayflow": {
        "label": "Overload, RelayFlow retries", "source": "stand-in", "retrieval": "A", "seed": 21,
        "arrival": {"kind": "steady", "rpm": 45, "duration_s": 120},
        "mock": {}, "scheduler": {},
        "notes": "Client limiter + jittered exponential backoff + Retry-After + retry budget + deadline.",
    },
}


def arrivals(sc: dict) -> list[float]:
    """Deterministic arrival offsets (seconds) for a scenario."""
    rng = random.Random(sc["seed"])
    a = sc["arrival"]

    def steady(rpm: float, dur: float) -> list[float]:
        period = 60.0 / rpm
        n = int(dur / period)
        return [max(0.0, i * period + rng.uniform(-0.2, 0.2) * period) for i in range(n)]

    if a["kind"] == "steady":
        return sorted(steady(a["rpm"], a["duration_s"]))
    if a["kind"] == "burst":
        base = steady(a["base_rpm"], a["duration_s"])
        burst = [a["burst_at_s"] + rng.uniform(0, a["burst_window_s"]) for _ in range(a["burst_n"])]
        return sorted(base + burst)
    raise ValueError(a["kind"])


def questions_for(sc: dict, n: int) -> list[str]:
    rng = random.Random(sc["seed"] + 1000)
    out: list[str] = []
    while len(out) < n:
        block = POOL[:]
        rng.shuffle(block)
        out.extend(block)
    return out[:n]


def write_configs() -> list[Path]:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for key, sc in SCENARIOS.items():
        full = {"key": key, **sc, "mock_base": BASE_MOCK, "scheduler_base": BASE_SCHED, "pool": POOL}
        full["n_requests"] = len(arrivals(sc))
        p = CONFIG_DIR / f"scenario_{key}.json"
        p.write_text(json.dumps(full, indent=2), encoding="utf-8")
        paths.append(p)
    return paths


def load(key: str) -> dict:
    return json.loads((CONFIG_DIR / f"scenario_{key}.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ns = ap.parse_args()
    if ns.write:
        for p in write_configs():
            print("wrote", p.relative_to(ROOT_DIR))
    else:
        for k, sc in SCENARIOS.items():
            print(k, len(arrivals(sc)), "requests")
