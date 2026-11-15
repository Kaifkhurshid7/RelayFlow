# Conversation history (user prompts, verbatim)

Source: Claude Code session transcript `c34d7205-415a-4f6c-bb9d-28bed2ba1b8f.jsonl`. 10 human messages, in order.
Credentials that appeared in the chat were redacted by `scripts/export_conversation.py`.
Messages the user pasted from elsewhere are shown as received. Assistant replies are NOT reproduced
here; see `docs/HOW_I_WORKED.md` for the decisions, mistakes and corrections, each tied to a file or
test in this repository.

## Prompt 1  (2026-11-14T02:19:09.611Z)

```text
<pasted_content id="5cbd">
# MASTER PROJECT PROMPT — RELAYFLOW
## Building a Resilient, Rate-Aware, Multi-Stage AI Research Assistant

Act as a Senior AI Systems Engineer, Backend Architect, Distributed Systems Engineer, Performance Engineer, and Technical Researcher.

I want you to help me design, implement, test, benchmark, document, and prepare a complete professional submission for the assignment titled **"The Relay"**.

The project name is **RelayFlow**.

This is not a basic chatbot or a simple RAG application. It is an engineering project focused on building a reliable multi-stage AI pipeline that continues operating intelligently under API rate limits, traffic bursts, latency spikes, tool failures, token constraints, and daily quota exhaustion.

Your responsibility is to help me build an actual working system, conduct real experiments, collect reproducible measurements, analyse the results, and prepare a technically defensible submission.

Do not merely generate code that looks correct. Every important component must be implemented, executed, tested, and verified.

---

# 1. PRIMARY OBJECTIVE

Build RelayFlow, a Python-based AI research assistant that answers engineering questions using exactly three specified research papers.

The system must:

1. Understand incoming questions.
2. Retrieve relevant passages from the research library.
3. Decide how to answer based on question complexity, evidence availability, and resource availability.
4. Generate answers using a real hosted LLM through Groq's free tier.
5. Use tools as part of the processing pipeline.
6. Verify factual claims and citations before returning answers.
7. Handle model provider rate limits and failures.
8. Support streaming responses.
9. Manage request queues and concurrency.
10. Retry intelligently without creating retry storms.
11. Degrade gracefully when resources are constrained.
12. Simulate realistic provider behaviour using a locally controlled stand-in model service.
13. Benchmark performance under different traffic patterns.
14. Record every request and its outcome in results.csv.
15. Produce a professional two-page technical report with all required tables.

The final implementation must be understandable, reproducible, modular, and suitable for a technical interview.

Do not overengineer. Prefer simple, explicit, measurable engineering decisions over unnecessary frameworks.

# 2. TECHNOLOGY STACK

Use the following technologies unless a strong technical reason requires a documented alternative.

- Python 3.11+
- FastAPI for API services
- Uvicorn for serving FastAPI
- Groq Python SDK for hosted LLM calls
- PyMuPDF for PDF extraction
- BM25 for lexical retrieval
- Sentence Transformers for optional semantic retrieval or reranking
- asyncio for asynchronous orchestration
- httpx for asynchronous HTTP calls
- Pydantic for validation and configuration
- pytest for testing
- pandas for experiment analysis
- Matplotlib for charts
- Locust or a custom asyncio-based load generator
- python-dotenv for environment configuration

Use direct Python orchestration instead of depending on a large agent framework.

The project must run locally with clear installation and execution instructions.

# 3. EXACT RESEARCH LIBRARY

The assistant must answer exclusively from these three exact paper versions.

Download and preserve the original PDFs.

1. Attention Is All You Need
   arXiv ID: 1706.03762
   Required version: v7
   URL: https://arxiv.org/pdf/1706.03762v7

2. Moshi: a speech-text foundation model for real-time dialogue
   arXiv ID: 2410.00037
   Required version: v1
   URL: https://arxiv.org/pdf/2410.00037v1

3. Dynamic Chunking for End-to-End Hierarchical Sequence Modeling
   arXiv ID: 2507.07955
   Required version: v2
   URL: https://arxiv.org/pdf/2507.07955v2

Requirements:

- Download the exact requested versions.
- Verify that each file corresponds to the correct arXiv version.
- Store the original PDFs locally.
- Extract text while retaining page numbers and section information wherever possible.
- Do not silently substitute the latest version of a paper.
- Preserve document metadata.
- Handle equations, tables, headings, references, and multi-column layouts carefully.
- Do not treat references and bibliographies as evidence for claims made in the papers unless explicitly justified.

Suggested directory:

data/
    papers/
        attention_is_all_you_need_v7.pdf
        moshi_v1.pdf
        dynamic_chunking_v2.pdf
    extracted/
    chunks/

If an exact version cannot be downloaded or verified, stop and report the problem rather than silently substituting another version.

# 4. SYSTEM ARCHITECTURE

Implement a multi-stage pipeline with at least five meaningful processing stages.

Recommended architecture:

User Request
    |
    v
Request Admission Controller
    |
    v
Question Understanding
    |
    v
Retrieval Engine
    |
    v
Decision and Resource Scheduler
    |
    v
LLM Generation
    |
    v
Citation and Evidence Verification
    |
    v
Response Streaming
    |
    v
Request Logging and Metrics

The system must contain:

- At least four distinct processing steps.
- At least one real tool call.
- At least one conditional branch.
- At least one loop.
- At least one document retrieval operation.
- At least one streaming response path.

Implement these modules:

1. request_manager.py
2. question_analyzer.py
3. paper_ingestion.py
4. retrieval.py
5. decision_engine.py
6. groq_client.py
7. mock_provider.py
8. scheduler.py
9. verification.py
10. pipeline.py
11. metrics.py
12. benchmark.py

The actual module structure can be improved if necessary, but maintain separation of responsibilities.

# 5. QUESTION UNDERSTANDING

Build a question analysis stage that identifies:

- Main topic.
- Relevant paper or papers.
- Question complexity.
- Whether multiple passages are required.
- Whether the question is answerable from the library.
- Expected retrieval depth.
- Whether a clarification or refusal is necessary.

Use a model call where appropriate.

Include a lightweight fallback when model capacity is unavailable.

Do not make unsupported assumptions about what a question means.

# 6. RETRIEVAL ENGINE

Build a proper document retrieval system.

Initial implementation:

- Extract paper text.
- Split text into configurable chunks.
- Preserve paper ID, paper title, version, page number, section heading, and chunk ID.
- Generate BM25 indexes.
- Retrieve relevant passages.
- Optionally use semantic similarity.
- Optionally rerank retrieved passages.

Test at least three retrieval configurations:

Configuration A:
- Chunk size: approximately 300 tokens.
- Retrieve 3 passages.
- No reranking.

Configuration B:
- Chunk size: approximately 600 tokens.
- Retrieve 4 passages.
- No reranking.

Configuration C:
- Chunk size: approximately 600 tokens.
- Retrieve 6 passages.
- Reranking enabled.

These are starting configurations, not mandatory optimal values. Adjust them only after measuring.

Measure:

- Retrieval relevance.
- Correctly supported answers.
- Incorrect answers.
- Correct refusals.
- Retrieval and generation token usage.
- Time to first token.
- Total response time.

Use the same evaluation questions across configurations.

Never allow the LLM to answer from general knowledge when the research papers do not contain sufficient evidence.

# 7. GROQ REAL MODEL INTEGRATION

Use the actual Groq hosted inference service with a personal free-tier account.

Official resources:

https://console.groq.com/docs/rate-limits

https://console.groq.com/docs/models

https://console.groq.com/docs/api-reference

Implementation requirements:

- Use environment variables for credentials.
- Never hardcode API keys.
- Never print API keys in logs.
- Use a configurable model name.
- Support asynchronous requests.
- Support streaming where available.
- Capture provider response headers.
- Capture token usage.
- Capture HTTP status codes.
- Capture provider error messages.
- Capture request duration.
- Capture time to first token.
- Capture total generation duration.

Create a .env.example file:

GROQ_API_KEY=your_key_here
GROQ_MODEL=your_model_here
PROVIDER_MODE=mock

Do not invent provider limits.

Research the current official documentation and record the applicable limits for the selected model and free plan.

Investigate:

- Requests per minute.
- Tokens per minute.
- Requests per day.
- Tokens per day, if applicable.
- Concurrent request restrictions, if documented.
- Model-specific limits.
- Response headers.
- HTTP 429 behaviour.
- Retry-After information, if present.
- Daily quota exhaustion behaviour.
- Recovery behaviour.

Distinguish clearly between documented limits and experimentally observed behaviour.

Conduct a modest number of real requests first.

Do not deliberately abuse or exhaust the free-tier account.

If a daily limit cannot safely be observed, report it as not measured rather than inventing an experiment.

Record the date of measurement because limits may change.

# 8. STAND-IN MODEL PROVIDER

Build a locally hosted FastAPI service that simulates an external model provider.

This is a critical part of the assignment.

The stand-in must support:

1. Normal responses.
2. Random latency.
3. Configurable slow responses.
4. Random failures.
5. HTTP 500 errors.
6. HTTP 429 rate-limit responses.
7. RPM limits.
8. TPM limits.
9. Daily quota exhaustion.
10. Concurrent request constraints.
11. Increasing latency under load.
12. Streaming responses.
13. Partial response delivery.
14. Request timeouts.
15. Retry-related scenarios.

Provide configurable parameters through environment variables or a configuration file.

Example:

MOCK_BASE_LATENCY_MS=500
MOCK_LATENCY_JITTER_MS=200
MOCK_MAX_RPM=30
MOCK_MAX_TPM=30000
MOCK_MAX_DAILY_REQUESTS=1000
MOCK_MAX_CONCURRENCY=10
MOCK_FAILURE_RATE=0.02
MOCK_OVERLOAD_FACTOR=1.5
MOCK_STREAMING_ENABLED=true

These are example settings, not claims about Groq.

Calibrate the mock provider using actual measurements from the real service.

Document:

- What was measured.
- How the parameters were selected.
- Which characteristics were approximated.
- Where the simulation differs most from Groq.

Do not claim the stand-in reproduces Groq's internal infrastructure.

The mock must allow experiments that cannot safely be run against the real free-tier provider.

# 9. SCHEDULER AND RESOURCE MANAGEMENT

Implement a scheduler that controls incoming workload.

Required features:

- Bounded asynchronous queue.
- Configurable concurrency.
- Request admission control.
- RPM tracking.
- TPM tracking.
- Daily budget tracking.
- In-flight request tracking.
- Queue waiting-time measurement.
- Request prioritisation.
- Timeout handling.
- Graceful overload behaviour.

Use a token-bucket or another justified rate-limiting algorithm.

Explain why you selected it.

The scheduler must distinguish:

- Requests accepted immediately.
- Requests waiting in the queue.
- Requests rejected before processing.
- Requests timing out.
- Requests that fail because of provider rate limits.

Avoid an unlimited queue.

Never silently discard requests.

When rejecting or deferring a request, record the reason.

# 10. DECISION ENGINE AND GRACEFUL DEGRADATION

Build a decision engine that chooses the appropriate processing strategy.

Consider:

- Question complexity.
- Retrieval relevance.
- Remaining token budget.
- Remaining request budget.
- Current queue length.
- Provider availability.
- Estimated response latency.

Implement multiple execution paths.

Normal path:
- Full retrieval.
- Appropriate model.
- Answer generation.
- Evidence verification.

Constrained path:
- Reduce unnecessary retrieval.
- Use fewer relevant passages.
- Use a cheaper suitable model if available and configured.
- Preserve evidence checking.

Exhausted-budget path:
- Do not make an API call.
- Return a transparent message explaining the temporary limitation.
- Do not invent an answer.
- Preserve the ability to retry when capacity becomes available.

Important principle:

Never trade factual correctness or citation integrity for lower latency.

A weaker answer may be used only when it remains adequately supported by the retrieved evidence.

# 11. CITATION AND ANSWER VERIFICATION

This is a mandatory reliability feature.

Every substantive answer must contain citations.

Citations must identify:

- Paper title.
- arXiv ID.
- Exact version.
- Section.
- Page number where available.

Implement a verification stage that checks:

- Whether cited passages exist.
- Whether cited paper versions are correct.
- Whether the answer is supported by retrieved evidence.
- Whether claims exceed the available evidence.
- Whether citations point to relevant passages.
- Whether the answer contains unsupported external knowledge.

If the papers do not answer a question, the system must explicitly say:

"The provided research library does not contain sufficient information to answer this question."

Do not manufacture citations.

Do not let a citation's mere presence count as proof of correctness.

Create tests for unsupported questions and misleading questions.

# 12. RETRY POLICY

Implement bounded retries.

Use:

- Exponential backoff.
- Random jitter.
- Retry-After where provided.
- Maximum retry attempts.
- Per-request deadlines.
- Retry classification.

Retry appropriate transient failures.

Do not repeatedly retry:

- Invalid requests.
- Authentication failures.
- Permanent errors.
- Daily quota exhaustion.

Treat per-minute rate limits and daily quota exhaustion differently.

Ensure that retries do not create retry storms.

Measure how retries affect:

- Total latency.
- Queue length.
- Provider load.
- Success rate.
- User waiting time.

Include at least one experiment where retrying makes the system worse.

Explain why it happens.

# 13. LOAD TESTING

Create a reproducible benchmarking framework.

Run at least these five scenarios:

A. STEADY LOAD

A relatively consistent incoming request rate.

B. BURST LOAD

A sudden increase in incoming requests over a short period.

C. SUSTAINED OVERLOAD

Incoming traffic remains above processing capacity.

D. REDUCED RPM LIMIT

Simulate a provider whose requests-per-minute limit is halved.

E. DAILY BUDGET EXHAUSTED

Simulate daily quota exhaustion at 3 PM.

Use real Groq requests only for modest, safe experiments.

Use the stand-in for aggressive load testing.

Do not represent simulated results as real Groq measurements.

For each scenario measure:

- Requests sent.
- Duration.
- Offered rate.
- Successful requests.
- Rate-limited requests.
- Timed-out requests.
- Given-up requests.
- Retry count.
- Peak queue length.
- Tokens per answer.
- Cost per 1,000 answers.
- TTFT p50.
- TTFT p95.
- TTFT p99.
- Total latency p50.
- Total latency p95.
- Total latency p99.

Measure TTFT using the arrival of the first actual generated response token.

Measure total time from request admission to completion.

Keep queue time, working time, and retry time as separate measurements.

Do not fabricate missing measurements.

# 14. RAW RESULTS FILE

Generate the mandatory results.csv.

It must have exactly these required columns:

request_id
scenario
source
start_time
first_token_ms
total_ms
status
retries
tokens_in
tokens_out
retrieval_setting
answer_grade

Allowed values:

source:
- real
- stand-in

status:
- ok
- rate_limited
- timeout
- error

answer_grade:
- correct
- wrong
- refused_correctly
- refused_wrongly

Every request must generate a record, including failures and refusals.

Use appropriate null/empty values when a metric does not exist.

Do not silently omit failed requests.

Keep additional diagnostic data in a separate event log or metrics file.

This auxiliary data should include:

- Queue-entry timestamp.
- Processing-start timestamp.
- First-token timestamp.
- Completion timestamp.
- Retry events.
- Provider response headers.
- HTTP status.
- Error classification.
- Retrieval duration.
- Generation duration.
- Verification duration.
- Queue waiting time.
- Tool execution duration.
- Retry waiting time.
- Model identifier.
- Request and response token counts.

All report statistics must be reproducible from the raw request records and the clearly documented auxiliary logs where necessary.

Do not put invented values in results.csv.

# 15. EVALUATION QUESTIONS

Write at least 20 original questions covering all three papers.

Include:

- Direct factual questions.
- Technical questions.
- Questions requiring multiple passages.
- Questions requiring cross-paper comparisons.
- Questions about limitations.
- Questions about experimental methods.
- Questions that cannot be answered from the library.
- Misleading questions.
- Questions containing terminology that could confuse a naive retriever.

For every question manually or systematically establish:

- Expected answer.
- Supporting paper.
- Supporting section/page.
- Whether it is answerable.
- Correct refusal criteria.

Grade responses:

correct
wrong
refused_correctly
refused_wrongly

Use at least 15 questions in the quality evaluation.

Do not use an LLM's self-assessment as the only measure of correctness.

Inspect evidence and citations.

# 16. RETRIEVAL QUALITY VERSUS COST

Run at least three retrieval configurations.

Produce Table 3 with:

- Configuration.
- Questions asked.
- Correct and cited percentage.
- Wrong percentage.
- Correctly refused percentage.
- Tokens per answer.
- TTFT p50 and p95.
- Total time p50 and p95.

Select the best configuration based on measured quality and resource usage.

Explain the trade-offs.

Also explain what happens to retrieval depth when the remaining provider budget becomes low.

# 17. THREE WAYS A NAIVE RETRIEVER CAN BE MISLED

Identify three concrete retrieval failure modes.

Consider:

1. Similar terminology appearing in different papers.
2. Important information split across distant sections or pages.
3. Relevant-looking passages that do not actually support the requested claim.

For each failure mode:

- Explain why it occurs.
- Give a specific example question.
- Show what a naive retriever might return.
- Explain why that passage is insufficient.
- Describe how RelayFlow handles the issue.

Base the examples on the actual papers.

# 18. LATENCY ANALYSIS

Choose at least one genuinely slow request.

Break its latency into:

- Queue waiting.
- Processing and tool execution.
- Model generation.
- Retry waiting.
- Verification.
- Total time.

Explain whether the main delay happened before or after the first token.

Support the analysis using timestamps and event logs.

Do not estimate these values from memory.

# 19. LIMIT-HANDLING ANALYSIS

Explain separately:

Per-minute limit:
- What happens to new requests?
- Which requests wait?
- Which requests are rejected?
- How does Retry-After influence behaviour?
- What happens to already running requests?

Per-day limit:
- What happens when the quota is exhausted?
- Can requests continue using a fallback provider?
- Which requests receive a transparent refusal?
- How are users informed?
- When can processing resume?

Explain who gets delayed, who gets dropped, and who receives a weaker answer.

Never suggest that the system can bypass provider limits.

# 20. THE 3 PM DAILY-BUDGET SCENARIO

Describe exactly what a waiting user sees when the daily provider budget is exhausted at 3 PM.

The response must be professional, honest and understandable.

It should explain:

- That the external inference quota is exhausted.
- That the system cannot safely generate a new answer at the moment.
- That the request has not been silently discarded.
- Whether it will be queued or require resubmission.
- When the user can reasonably try again, if known.

Do not promise an exact recovery time without evidence.

# 21. VAANI FEEDBACK

Visit:

https://www.vaani.dev/

Explore the actual website.

Write a few honest lines describing:

- What stood out.
- What was confusing.
- One improvement you would suggest.

Do not invent website features.

Do not write generic praise.

Base the feedback on what you actually observe.

# 22. FINAL TWO-PAGE TECHNICAL REPORT

Produce a concise professional report of no more than two pages, excluding code, raw results and conversation history.

Use these headings in exactly this order:

1. What I built and why.
2. The real service and my stand-in.
3. Results.
4. Where the time goes.
5. When a limit is hit.

Then add:

6. Retrieval: quality versus cost.
7. How this library will mislead a naive retriever.

Then continue with:

8. Retries.
9. What the person sees.
10. What I'm unsure about, what I left out, what I learned.
11. Your feedback on Vaani.

Finish with:

12. How I worked with the assistant.

Preserve all requested topics and required content from the original assignment.

Include all mandatory results tables.

Do not exceed two pages through unnecessarily large headings, excessive whitespace, or decorative formatting.

The report must be concise but technically meaningful.

# 23. REQUIRED RESULTS TABLES

Table 1: Limits found in the real service.

Columns:

Limit | Value you found | How you found it | Confidence

Rows:

- Requests per minute.
- Tokens per minute.
- Requests per day.
- Tokens per day.
- Requests in flight at once.
- What it returns when a limit is hit.
- Time until it recovers.

Table 2a: Latency.

Include one row per scenario:

- Steady.
- Burst.
- Sustained overload.
- Per-minute limit halved.
- Daily budget exhausted.

Columns:

Scenario | Real or stand-in | Requests sent | Duration (s) | Offered rate (per min) | First token p50/p95/p99 (ms) | Total time p50/p95/p99 (ms)

Table 2b: Outcomes.

Use the same scenario rows.

Columns:

Scenario | Succeeded % | Rate-limited % | Timed out or given up % | Retries per request | Peak queue length | Tokens per answer | Cost per 1,000 answers

Table 3: Retrieval settings.

Include at least three configurations.

Columns:

Setting | Questions asked | Correct and cited % | Wrong % | Correctly refused % | Tokens per answer | First token p50/p95 (ms) | Total time p50/p95 (ms)

Rules:

- Every percentage must have one decimal place.
- All time measurements must be in milliseconds.
- Use not measured where a value was not measured.
- Do not invent missing values.
- Ensure all calculated statistics match the raw data.
- Explain sample sizes and percentile calculation methods.

# 24. HOW I WORKED WITH THE ASSISTANT

Maintain a genuine record of the development conversation.

Include:

- Actual prompts used.
- Important implementation decisions.
- Mistakes discovered.
- Corrections made.
- Suggestions rejected.
- Decisions made by me rather than blindly accepting AI output.

Do not fabricate conversation history.

Do not manufacture a story about catching mistakes.

At the end, help me identify:

1. The three prompts that mattered most and why.
2. One actual AI-generated mistake I caught and how I verified it.
3. One design decision where I overruled the assistant.
4. Which parts of the submission are distinctly my own work.

Keep the actual development history available for export.

# 25. ENGINEERING QUALITY REQUIREMENTS

The implementation must satisfy the following:

- Clean modular architecture.
- Type hints.
- Meaningful docstrings.
- Structured logging.
- Centralised configuration.
- Proper exception handling.
- No hardcoded credentials.
- No fake measurements.
- No unnecessary dependencies.
- Reproducible random seeds for simulated experiments.
- Unit tests for critical components.
- Integration tests for the complete pipeline.
- Tests for rate limits.
- Tests for retries.
- Tests for streaming.
- Tests for timeout behaviour.
- Tests for citation verification.
- Tests for unsupported questions.
- Tests for daily quota exhaustion.

Include:

- README.md
- requirements.txt
- .env.example
- .gitignore
- results.csv
- Experiment configuration files
- Raw diagnostic logs
- Test suite
- Two-page report

Make sure the project can be run from a clean environment.

# 26. RECOMMENDED PROJECT STRUCTURE

RelayFlow/
│
├── app/
│   ├── main.py
│   ├── config.py
│   ├── api/
│   ├── pipeline/
│   ├── retrieval/
│   ├── providers/
│   ├── scheduler/
│   ├── verification/
│   └── monitoring/
│
├── data/
│   ├── papers/
│   ├── extracted/
│   └── indexes/
│
├── mock_service/
│   ├── main.py
│   ├── simulator.py
│   └── limits.py
│
├── benchmarks/
│   ├── load_generator.py
│   ├── scenarios.py
│   └── analysis.py
│
├── evaluation/
│   ├── questions.json
│   ├── ground_truth.json
│   └── evaluator.py
│
├── tests/
│
├── experiments/
│   ├── configurations/
│   ├── logs/
│   └── raw/
│
├── results/
│   ├── results.csv
│   ├── event_logs.jsonl
│   └── charts/
│
├── report/
│   └── final_report.pdf
│
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md

You may modify the structure if doing so improves maintainability.

# 27. IMPLEMENTATION WORKFLOW

Work incrementally.

Do not generate the entire application in one enormous code dump.

Follow these phases.

PHASE 1 — PROJECT FOUNDATION

- Create the repository.
- Configure Python environment.
- Set up dependencies.
- Create modular structure.
- Implement central configuration.
- Verify installation.

PHASE 2 — PAPER INGESTION AND RETRIEVAL

- Download exact paper versions.
- Verify version identifiers.
- Extract PDF content.
- Preserve page and section metadata.
- Implement chunking.
- Implement BM25.
- Test retrieval against sample questions.

PHASE 3 — WORKING AI PIPELINE

- Integrate Groq.
- Implement question understanding.
- Implement retrieval.
- Implement generation.
- Implement evidence verification.
- Implement citations.
- Implement streaming.
- Test complete question-to-answer flow.

PHASE 4 — PROVIDER SIMULATION

- Implement FastAPI stand-in.
- Add configurable latency.
- Add failures.
- Add RPM and TPM enforcement.
- Add daily quota.
- Add concurrency limits.
- Add streaming.
- Add overload behaviour.
- Calibrate from real measurements.

PHASE 5 — SCHEDULER AND RESILIENCE

- Implement bounded queue.
- Implement token bucket.
- Implement admission control.
- Implement concurrency management.
- Implement retries.
- Implement backoff and jitter.
- Implement fallback and refusal paths.

PHASE 6 — EXPERIMENTS

- Conduct modest real-provider measurements.
- Run five load scenarios.
- Run three retrieval configurations.
- Evaluate at least 15 questions.
- Save every request.
- Generate diagnostic event logs.

PHASE 7 — RESULTS AND ANALYSIS

- Calculate percentiles.
- Calculate success and failure rates.
- Analyse queueing.
- Analyse retries.
- Analyse resource consumption.
- Calculate cost estimates.
- Validate all statistics against raw records.

PHASE 8 — FINAL SUBMISSION

- Prepare the two-page report.
- Include all required headings.
- Include the mandatory tables.
- Include actual Vaani feedback.
- Prepare reproducible instructions.
- Verify results.csv.
- Prepare conversation history.
- Review the complete submission.

# 28. RULES FOR WORKING WITH ME

I want you to act as an engineering collaborator, not an autonomous code generator.

Follow these rules:

1. Explain the purpose of each major implementation decision.
2. Generate production-quality, readable code.
3. Do not skip difficult requirements.
4. Do not claim that code works until it has been executed and tested.
5. Do not claim that experiments have been completed before actually running them.
6. Never fabricate Groq limits, latency, token usage, answer quality, or costs.
7. Never fabricate results to make the report look impressive.
8. Never substitute simulated measurements for real measurements without explicitly identifying them.
9. Never expose my API key.
10. Ask me for credentials only when required, and tell me to provide them through environment variables.
11. If a technical issue occurs, investigate it and propose a verified solution.
12. Keep the project within a reasonable 6–8 hour engineering effort.
13. Prioritise assignment requirements over unnecessary features.
14. Maintain a checklist of completed, partially completed and pending requirements.
15. Keep the repository organised throughout development.
16. Ensure the final report matches the actual experiment results.
17. Do not silently make major architecture changes.
18. Explain trade-offs clearly so I can defend them during an interview.

When something cannot be measured or implemented, document the limitation honestly.

# 29. FINAL ACCEPTANCE CHECKLIST

Before declaring the project complete, verify every item.

[ ] All three exact arXiv versions downloaded and verified.

[ ] PDF extraction implemented.

[ ] Retrieval implemented and tested.

[ ] At least five pipeline stages implemented.

[ ] At least one tool call implemented.

[ ] Conditional branching implemented.

[ ] Loop implemented.

[ ] Groq real API integration tested.

[ ] Actual provider limits investigated.

[ ] Actual latency measurements recorded.

[ ] Response headers and errors captured.

[ ] Stand-in provider implemented.

[ ] Streaming supported.

[ ] Configurable latency supported.

[ ] Failure simulation supported.

[ ] RPM and TPM limits supported.

[ ] Daily quota simulation supported.

[ ] Overload latency supported.

[ ] Bounded queue implemented.

[ ] Retry policy implemented.

[ ] Evidence verification implemented.

[ ] Five load scenarios completed.

[ ] Three retrieval configurations evaluated.

[ ] At least 15 questions evaluated.

[ ] TTFT and total latency calculated.

[ ] p50, p95 and p99 calculated where sample sizes permit.

[ ] Every request recorded in results.csv.

[ ] Auxiliary diagnostic logs preserved.

[ ] Tables match raw data.

[ ] Two-page report completed.

[ ] Vaani website actually inspected.

[ ] README and setup instructions completed.

[ ] Tests executed successfully.

[ ] No credentials committed.

[ ] Genuine conversation history preserved.

[ ] All claims in the report are defensible.

---

# 30. START NOW

Begin with PHASE 1.

First, inspect the current working directory and understand the available environment.

Then:

1. Present the proposed architecture briefly.
2. Identify any required credentials or external dependencies.
3. Create the initial project structure.
4. Set up the Python environment.
5. Implement the initial configuration and project entry points.
6. Run a basic verification test.
7. Show me the files created and the exact commands to run.

After Phase 1, continue to Phase 2 and subsequent phases.

Do not stop at planning or generating a README.

The final objective is a genuinely implemented, tested, measured and documented RelayFlow system that satisfies the original assignment.

**Engineering principle: Measure first. Design deliberately. Implement incrementally. Test everything. Report honestly.**
</pasted_content id="5cbd">

   but firstlty setup git igmore amd readme so that i can push into github and start building it
```

## Prompt 2  (2026-11-14T02:20:58.252Z)

```text
make the overall folder structure also to push
```

## Prompt 3  (2026-11-14T02:28:43.831Z)

```text
now start doing the project but never commit any changes striclty
```

## Prompt 4  (2026-11-14T02:42:08.018Z)

```text
<pasted_content id="5cbd">
GEMINI_API_KEY_1=AQ.[REDACTED]
GEMINI_API_KEY_2=AQ.[REDACTED]
GEMINI_API_KEY_3=AQ.[REDACTED]
GEMINI_MODEL=gemini-3.8-flash

GROQ_API_KEY=gsk_[REDACTED]
GROQ_API_KEY_2=gsk_[REDACTED]
GROQ_MODEL=openai/gpt-oss-120b
</pasted_content id="5cbd">

  use
```

## Prompt 5  (2026-11-14T02:43:19.765Z)

```text
test it whether working or not
```

## Prompt 6  (2026-11-14T02:49:02.155Z)

```text
is all done now?
```

## Prompt 7  (2026-11-14T02:49:25.113Z)

```text
see do all the phases compelte thiis work properly
```

## Prompt 8  (2026-11-15T02:25:30.348Z)

```text
tell me is all done now
```

## Prompt 9  (2026-11-15T02:27:38.533Z)

```text
do it fast
```

## Prompt 10  (2026-11-15T02:49:07.137Z)

```text
Base directory for this skill: C:\Users\REALME\AppData\Local\Temp\claude\bundled-skills\2.1.325\b46619e923c0c49d502c8d44c441d31b\dataviz

# Data Visualization

A chart is **read by people and executed by you**. This skill turns "make it look
good" into a procedure with checks, so the result is right by construction rather
than by taste.

**The method here is design-system-agnostic.** Nothing in the procedure, the form
heuristic, the six checks, or the mark specs is specific to one product. A design
system supplies a small set of *parameters* (its ramps, a categorical order, a
diverging pair, a status palette, a texture, its surfaces, its filter components);
the method consumes them unchanged. A **validated default palette** is the
reference instance, fully specified in `references/palette.md`. To target your
brand, read that file's structure and substitute its values - touch nothing else.

> The single most important habit: **the color part is computable, so compute it.**
> Never eyeball whether a palette is colorblind-safe - run `scripts/validate_palette.js`.

## The procedure - do these in order

Color comes LAST. Most bad charts pick colors first.

1. **Pick the form.** What is the data's job - magnitude, identity, polarity, a
   single headline, change-over-time? The job picks the chart type, and sometimes
   the answer is *not a chart* (a stat tile or hero number). -> `references/choosing-a-form.md`
2. **Assign color by the job it does.** Categorical (identity), sequential
   (magnitude), diverging (polarity), or status (state) - each has one rule.
   Assign categorical hues in fixed order, never cycled. -> `references/color-formula.md`
3. **VALIDATE the palette - run the script, don't reason about Delta E.**
   `node scripts/validate_palette.js "<hex,hex,...>" --mode light` (relative to
   this skill's base directory - or load it as `<script type="module">` in the
   chart's own page, where it reads
   `data-palette` off `<body>` and logs a `console.table` report). It returns
   pass/fail on the lightness band, chroma floor, adjacent-pair CVD separation,
   the normal-vision floor, and contrast. Fix anything that FAILs before continuing. Re-run for
   `--mode dark` with that mode's surface.
4. **Apply mark specs & spacers.** Thin marks, 4px rounded data-ends anchored to
   the baseline, 2px lines, >=8px markers, a 2px surface gap between fills (stacked
   segments and adjacent bars alike) and a 2px surface ring on overlapping marks,
   selective direct labels. -> `references/marks-and-anatomy.md`
5. **Add the hover layer - by default.** An HTML/SVG chart *is* interactive; ship
   a crosshair+tooltip on line/area and a per-mark hover tooltip on bar/dot/cell.
   The only form that skips it is a bare stat tile with no plot. Hit targets bigger
   than the mark; filters in one row above the charts. -> `references/interaction.md`
6. **Final accessibility pass.** For >= 2 series a legend is always present and <= 4
   are also direct-labeled (a single series needs no legend box - the title names
   it), so identity is never color-alone; a table view exists; dark mode is **selected** - its own
   steps from the same ramps, validated against the dark surface, not an automatic
   flip; texture is available for the CVD/print/forced-colors case.
   One exception to direct name labels: the tall segmented bar (*Figures* in `references/marks-and-anatomy.md`).
7. **Render it and look at it.** The validator checks color, not layout - open or
   screenshot the output and eyeball it for label collisions, geometry, and overflow
   before calling it done.

Then check the result against **`references/anti-patterns.md`** - it is the catalog
of what goes wrong. If your chart matches an entry, it's wrong.

## Non-negotiables (true in every design system)

- **Assign categorical hues in fixed order, never cycled.** A 9th series is never a
  generated hue - it folds into "Other," small multiples, or composite encoding.
- **One axis.** Never a dual-axis chart (two y-scales). Two measures of different
  scale -> two charts, small multiples, or indexed to a common base. *(This is the
  #1 chart mistake - see anti-patterns.)*
- **Color follows the entity, never its rank.** A filter that changes the series
  count must not repaint the survivors.
- **Sequential = one hue, light->dark. Diverging = two hues + a neutral gray
  midpoint.** Never a rainbow; never a hue at the diverging midpoint.
- **Run the validator before shipping any categorical palette.** CVD Delta E >= 8 is the
  target (OKLab ×100); 6-8 is a floor that is legal ONLY with secondary encoding. A
  normal-vision floor below 15 is a hard FAIL - full-color readers can't tell the
  pair apart; re-step it on the adjacent pairlist (secondary encoding does not excuse
  this one); under `--pairs all` cut series or facet instead - see check 4. A contrast WARN
  obligates visible labels or a table view - it is not dismissable.
- **Thin marks; a legend always present for >= 2 series (none for one), with
  selective direct labels (never a number on every point); recessive grid/axes.**
  One exception, legend kept: the tall segmented bar (*Figures* in `references/marks-and-anatomy.md`).
- **Text wears text tokens, never the series color** - values, labels, and legends
  stay in primary/secondary/muted ink; a colored mark beside them carries identity.
- **Status colors are reserved** (good/warning/serious/critical) and never reused
  for "series 4"; they ship with an icon + label, never color alone.

## Plugging in a design system

The method is invariant; only these parameters change per system. The reference
instance - every value filled in - is `references/palette.md`.

| Parameter | What the system provides |
|---|---|
| **Ramps** | the hue scales (named steps) the palette draws from |
| **Categorical theme** | the fixed hue order (a named theme); default + alternates |
| **Sequential hue** | the default single hue for magnitude |
| **Diverging pair** | two warm/cool poles + a neutral midpoint |
| **Status palette** | good / warning / serious / critical - steps distinct from categorical, except that dark-mode good is the series green |
| **Texture fill** | one directional hand-drawn fill, used at 45° / 135° |
| **Surfaces** | light & dark chart-surface colors (the validator needs these) |
| **Filter controls** | date-range & dimension controls (behavioral spec in `interaction.md`) |

To onboard a new system: fill those rows, feed its ramps to the validator, and let
it snap each slot to the nearest passing step. Structure and rules stay as written.

## Reference files

| File | What it answers |
|------|-----------------|
| `references/choosing-a-form.md` | Which chart type / is it even a chart? |
| `references/color-formula.md` | The four jobs, the six checks, snap-to-passing |
| `references/marks-and-anatomy.md` | Mark specs, spacers, labels, figures, hero number |
| `references/interaction.md` | Tooltips & hover, filters & time ranges |
| `references/components.md` | The pieces a chart is made of - build each in plain HTML |
| `references/anti-patterns.md` | **What goes wrong - check every chart against this** |
| `references/palette.md` | **The reference palette instance** - every parameter, filled in; swap for your brand's |
| `scripts/validate_palette.js` | Runnable six-checks validator (run it; don't eyeball) |
```
