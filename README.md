# RelayFlow

**Author: Kaif Khurshid**

A resilient, rate-aware, multi-stage AI research assistant. It answers engineering questions from
exactly three arXiv papers, cites its evidence, refuses what the papers do not support, and keeps
working under rate limits, bursts, latency spikes and daily-quota exhaustion.

Real provider: **Groq free tier, `openai/gpt-oss-120b`**. A local FastAPI **stand-in** provider
reproduces limits and faults for load tests that must not be run against the real account.
Stand-in results are always labelled `stand-in` and are never presented as Groq measurements.

## Research library (exact versions, verified by the arXiv stamp on page 1)

| Paper | arXiv ID | Version |
|---|---|---|
| Attention Is All You Need | 1706.03762 | v7 |
| Moshi: a speech-text foundation model for real-time dialogue | 2410.00037 | v1 |
| Dynamic Chunking for End-to-End Hierarchical Sequence Modeling | 2507.07955 | v2 |

## Setup

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env          # then edit: GROQ_API_KEY, GROQ_MODEL, PROVIDER_MODE
python -m app.retrieval.download           # downloads + verifies the three PDFs
python -m app.retrieval.paper_ingestion    # extraction + chunk files in data/extracted
pytest -q                                  # ~90 tests, no network, no real tokens
```
Keys live only in `.env` (git-ignored). `PROVIDER_MODE=mock` needs no key.

## Run

```powershell
uvicorn mock_service.main:app --port 8001      # the stand-in provider (MOCK_* env vars configure it)
uvicorn app.main:app --port 8000               # RelayFlow API: POST /ask, POST /ask/stream (SSE)
python -m evaluation.retrieval_eval --save     # offline retrieval comparison, no API calls
python -m evaluation.run_eval --mode groq --name quality_real --ids Q01,Q03 --configs A,B,C --max-tokens 125000
python -m benchmarks.scenarios --write         # write experiments/configurations/scenario_*.json
python -m benchmarks.load_generator steady burst overload rpm_halved daily_exhausted --mode mock
python -m benchmarks.load_generator real_steady real_burst --mode groq     # modest real runs
python -m benchmarks.calibrate quality_real real_steady real_burst         # fit stand-in latency
python -m benchmarks.analysis build            # results/results.csv, event_logs.jsonl, tables.json
```
The real-Groq commands spend free-tier tokens (documented limit 200K tokens/day); each takes a cap.

## Pipeline

```
request -> admission (bounded priority queue, deadline) -> question analysis (rules; optional LLM assist)
        -> retrieval (BM25 + stemming, per-paper loop, scope widening)   [tools: search_library, evidence_coverage]
        -> decision engine: normal | constrained | exhausted | unavailable | refuse_no_evidence | clarify
        -> ProviderGateway (client token buckets, retries, daily-quota state) -> Groq or stand-in (streaming)
        -> verification [tool: verify_answer] (+ bounded repair loop) -> verified answer streamed to the caller
        -> one event per request (JSONL) -> results.csv
```
Key design points (all covered by tests in `tests/`):
* **Citations cannot be fabricated**: the model emits `[S#]` tags; the system expands them from retrieved-passage
  metadata (paper, arXiv id, exact version, section, page). Unknown tags fail verification. Each answer sentence must
  be supported by the passages it cites (including every number), and a question that names a paper requires a
  citation from that paper.
* **Refusal**: no evidence -> exact sentence "The provided research library does not contain sufficient information
  to answer this question." Retrieval score alone is *not* a refusal signal (a lexical hit such as "NVIDIA P100" can
  look relevant); the evidence-coverage gate, the model and verification all participate.
* **Limits**: client token buckets mirror the provider (the real headers show continuously refilling buckets),
  AIMD adaptation on 429, Retry-After honoured, jittered exponential backoff, a retry budget (anti retry-storm),
  per-request deadlines, a daily-quota state that stops calling the API and returns a transparent message.

## Layout

```
app/            config, pipeline, retrieval, providers, scheduler, verification, monitoring, api
mock_service/   stand-in provider (limits.py, simulator.py, main.py)
benchmarks/     scenarios, load_generator, calibrate, analysis
evaluation/     questions.json, ground_truth.json, evaluator, run_eval, retrieval_eval
experiments/    configurations/ (scenario + calibration JSON), observations/ (real Groq probes)
results/        results.csv, event_logs.jsonl, tables.json, runs/<run>/ (raw events), charts/
report/         final report
docs/           conversation_history.md, HOW_I_WORKED.md
tests/          unit + integration tests
```

## results.csv

Exactly: `request_id, scenario, source, start_time, first_token_ms, total_ms, status, retries, tokens_in,
tokens_out, retrieval_setting, answer_grade`. `source` is `real` or `stand-in`; `status` is `ok | rate_limited |
timeout | error`. Empty cells mean the metric does not exist for that request (a refusal made without an API call
has no first token; a capacity message has no grade). `answer_grade` is produced by `evaluation/evaluator.py`
from ground truth (key facts, cited paper/pages, facts present in the cited text) - never by the model's opinion -
and is empty when the system produced neither an answer nor an evidence-based refusal. Diagnostic detail
(timestamps, headers, attempts, per-stage timings) is in `results/event_logs.jsonl`.

<!-- RESULTS:START -->

## Results

Full two-page report: [`report/final_report.pdf`](report/final_report.pdf). All numbers below are computed from `results/results.csv` / `results/event_logs.jsonl` by `benchmarks/analysis.py`. **Real** rows are Groq free tier (`openai/gpt-oss-120b`); **stand-in** rows come from the local simulator and are not Groq measurements.

### Table 1. Limits found in the real service

| Limit | Value you found | How you found it | Confidence |
|---|---|---|---|
| Requests per minute | 30 | Docs; never reached | Medium (docs only) |
| Tokens per minute | 8,000, charged at admission as prompt + max_tokens | Docs; `x-ratelimit-limit-tokens`=8000; real 429s with remaining-tokens 817-1,779 | High |
| Requests per day | 1,000, refilling ~86.4 s/request (inferred) | Docs; `limit-requests`=1000; `reset-requests` 1m26.4s | Value high; refill medium |
| Tokens per day | 200,000 | Docs only; no header; stayed under it, not tested | Low (not measured) |
| Requests in flight | Not documented; 6 simultaneous accepted | Real burst run | Low |
| Returned when a limit is hit | HTTP 429 + JSON error + `retry-after` (s) only on 429; `x-ratelimit-*` on every reply | 4 real 429s (Retry-After 1, 2, 7, 10 s) | High |
| Time until it recovers | Per-minute: the Retry-After (retries succeeded). Daily: not measured | Real burst run | Minute: medium; day: none |

### Table 2a. Latency (ms, successful requests; n/m = sample too small for that percentile)

| Scenario | Real or stand-in | Requests sent | Duration (s) | Offered rate (per min) | First token p50/p95/p99 (ms) | Total time p50/p95/p99 (ms) |
|---|---|---|---|---|---|---|
| Steady | real | 8 | 107 | 4 | 885/n/m/n/m | 1172/n/m/n/m |
| Burst | real | 6 | 20 | 360 | 642/n/m/n/m | 883/n/m/n/m |
| Steady | stand-in | 60 | 148 | 24 | 761/1128/n/m | 805/1533/n/m |
| Burst | stand-in | 85 | 144 | 34 | 5789/44666/n/m | 6899/47941/n/m |
| Sustained overload | stand-in | 180 | 238 | 60 | 12293/57439/58625 | 13938/57325/58607 |
| Per-minute limit halved | stand-in | 60 | 199 | 24 | 1283/51494/n/m | 2000/51998/n/m |
| Daily budget exhausted | stand-in | 80 | 237 | 20 | 711/1205/n/m | 711/1406/n/m |

### Table 2b. Outcomes (disjoint buckets; cost is notional list price, the free tier bills $0)

| Scenario | Succeeded % | Rate-limited % | Timed out or given up % | Retries per request | Peak queue length | Tokens per answer | Cost per 1,000 answers |
|---|---|---|---|---|---|---|---|
| Steady (real) | 100.0 | 0.0 | 0.0 | 0.00 | 1 | 1602.4 | $0.31 |
| Burst (real) | 100.0 | 0.0 | 0.0 | 0.67 | 1 | 1157.8 | $0.22 |
| Steady (stand-in) | 100.0 | 0.0 | 0.0 | 0.00 | 1 | 1324.0 | $0.28 |
| Burst (stand-in) | 89.4 | 0.0 | 10.6 | 0.02 | 40 | 1233.1 | $0.26 |
| Sustained overload (stand-in) | 72.2 | 3.3 | 24.4 | 0.02 | 40 | 1214.0 | $0.26 |
| Per-minute limit halved (stand-in) | 95.0 | 0.0 | 5.0 | 0.03 | 10 | 1324.8 | $0.28 |
| Daily budget exhausted (stand-in) | 75.0 | 25.0 | 0.0 | 0.01 | 1 | 1002.9 | $0.22 |

### Table 3. Retrieval settings (real Groq; 16 questions each; % of all questions asked; dagger = p95 from n<20)

| Setting | Questions asked | Correct and cited % | Wrong % | Correctly refused % | Tokens per answer | First token p50/p95 (ms) | Total time p50/p95 (ms) |
|---|---|---|---|---|---|---|---|
| A: 300 tok x3, no rerank | 16 | 50.0 | 6.3 | 31.3 | 1419 | 672/n/m | 844/n/m |
| B: 600 tok x4, no rerank | 16 | 56.3 | 6.3 | 31.3 | 3152 | 788/1084† | 1187/8006† |
| C: 600 tok x6, rerank | 16 | 50.0 | 6.3 | 31.3 | 4048 | 881/n/m | 1204/n/m |

**Choice: B.** Best measured quality per token (C costs ~28% more tokens for no gain). After the run, two verifier bugs on real model output were found and fixed; replaying the stored drafts offline (`results/verifier_replay.json`) lifts B and C to 62.5% correct and leaves A at 50.0%. 11 answerable questions are too few to rank configurations reliably, and settings were tuned on the same questions.

### Key findings

* **Where the time goes:** the slowest request (stand-in overload) took 59.1 s: 47.0 s queue + 11.0 s rate-limit wait + 1.09 s model. All of it was before the first token.
* **Retries can make things worse:** same overloaded provider, no retries sent 96 calls (90.0% ok); naive immediate retries sent 159 (+66%, 68 rejected, 87.8% ok); RelayFlow's policy sent 99 (100.0% ok) but users waited (p95 first token 27740 ms). One run each, small effect on success.
* **Daily budget gone (compressed day, 15:00 = t=180 s):** 75% served, 25% got a transparent message; the provider saw only 3 daily rejections for 80 requests.
* **Real-model surprises, all fixed with tests:** citations as `【S1】`, `[S3†L1-L5]`, "100 000" thousands separators; ~1.5 tokens/word instead of 1.3.

Charts: `results/charts/queue_length.png`, `results/charts/retry_comparison.png`.

<!-- RESULTS:END -->

## Honest limitations

* Quality is measured on a small question set (16 questions in the real 3-configuration run). Retrieval settings
  and the coverage gate were tuned on the same 31 questions; there is no held-out set.
* The stand-in is calibrated for latency/tokens only. Failure rate, load-dependent slowdown and concurrency limits
  are assumptions, and the daily limit was never exhausted on the real service.
* Stand-in TPM is raised to 60,000 (real: 8,000) so that the request limit binds; labelled in every table.
* Token counts use 1.5 tokens/word for chunk sizing and request estimates; the provider's own usage is what is reported.

I built this with Claude Code as my coding assistant; how that went, including what went wrong, is in `docs/HOW_I_WORKED.md`, and my prompts are in `docs/conversation_history.md`.

Nothing here has been committed yet; I will commit when I have reviewed it. Never commit `.env`.
