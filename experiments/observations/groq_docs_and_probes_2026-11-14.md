# Groq: documented limits vs. observed behaviour (measured 2026-11-14)

Model: `openai/gpt-oss-120b`. Account: one personal free-tier account (org-level limits).

## Documented (https://console.groq.com/docs/rate-limits, fetched 2026-11-14)
| Limit | Documented value |
|---|---|
| RPM | 30 |
| RPD | 1,000 |
| TPM | 8,000 |
| TPD | 200,000 |
| Concurrency | not documented |
| Daily reset time | not documented |
| 429 behaviour | HTTP 429; `retry-after` header (seconds) only on 429 |
Caveat: the docs page does not label which plan tab each number comes from; its text says the
values shown are the Developer-plan base limits and to check the console for the org's own limits.
Pricing page: $0.15 / 1M input tokens, $0.60 / 1M output tokens (notional cost only; free tier bills $0).

## Observed (real requests, ~6 tiny calls, no limit hit)
* `x-ratelimit-limit-requests: 1000`, `x-ratelimit-limit-tokens: 8000` -> match the documented RPD / TPM.
* `x-ratelimit-reset-tokens` after a request == tokens_charged / (8000/60 per s). E.g. 128 tokens -> 0.96 s,
  462 tokens -> 3.465 s. => TPM behaves as a continuously refilling token bucket.
* Tokens charged at admission ~= prompt_tokens + requested max_tokens (73+50 -> 128 used; 81+400 -> 462 used incl. refill).
* `x-ratelimit-reset-requests: 1m26.4s` after one request == 86,400 s / 1000. => RPD looks like a continuously
  refilling bucket (one request recovers every 86.4 s). INFERRED from headers; the daily limit was NOT exhausted.
* `openai/gpt-oss-120b` is a reasoning model: with max_tokens=30, 28 tokens were hidden reasoning and content was empty
  (finish_reason=length). `reasoning_effort="low"` cut reasoning to ~5 tokens. Streaming delivers `delta.reasoning`
  before `delta.content`; usage arrives in the final chunk (`x_groq.usage`).
* Latency (tiny prompts): ~0.65-1.0 s to first byte/token, whole 46-token answer in ~0.74 s.
* `x-groq-region: bom`.
