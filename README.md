# RelayFlow

A resilient, rate-aware, multi-stage AI research assistant. It answers engineering
questions from exactly three arXiv papers, and keeps working under API rate limits,
traffic bursts, latency spikes, and daily quota exhaustion.

> **Status: Phase 1 (project foundation) — work in progress.**
> Nothing below is claimed to work until it has been run and tested. See the checklist.

## Research library (exact versions)

| Paper | arXiv ID | Version |
|---|---|---|
| Attention Is All You Need | 1706.03762 | v7 |
| Moshi: a speech-text foundation model for real-time dialogue | 2410.00037 | v1 |
| Dynamic Chunking for End-to-End Hierarchical Sequence Modeling | 2507.07955 | v2 |

## Planned pipeline

```
Request -> Admission/Scheduler -> Question Analysis -> Retrieval (BM25)
        -> Decision Engine -> LLM (Groq or local stand-in)
        -> Citation/Evidence Verification -> Streaming response -> Metrics/CSV
```

Two providers sit behind one interface: the real **Groq** free tier, and a local
**FastAPI stand-in** with configurable latency, failures, RPM/TPM/daily limits and
streaming, used for aggressive load tests that must not be run against the real service.
Stand-in results are always labelled `stand-in` and never presented as Groq measurements.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env      # then edit .env; never commit it
```

Keep `GROQ_API_KEY` only in `.env` or your environment. `PROVIDER_MODE=mock` runs
without any key.

## Layout

```
app/          pipeline, retrieval, providers, scheduler, verification, monitoring
mock_service/ FastAPI stand-in provider
benchmarks/   load generator, scenarios, analysis
evaluation/   questions, ground truth, evaluator
tests/        unit + integration tests
data/         papers (PDFs not committed), extracted text, indexes
results/      results.csv, event_logs.jsonl, charts
report/       two-page technical report
```

## Checklist

- [ ] Phase 1 — Foundation (git, config, structure)
- [ ] Phase 2 — Paper ingestion and retrieval
- [ ] Phase 3 — Pipeline with Groq, verification, streaming
- [ ] Phase 4 — Stand-in provider
- [ ] Phase 5 — Scheduler and resilience
- [ ] Phase 6 — Experiments
- [ ] Phase 7 — Analysis
- [ ] Phase 8 — Report and submission

## License

TBD.
# RelayFlow
