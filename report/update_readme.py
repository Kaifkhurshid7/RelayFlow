"""Write the Results section of README.md from results/tables.json (same source as the report).

Run after `python -m benchmarks.analysis build`:  python report/update_readme.py
"""
from __future__ import annotations

import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
T = json.loads((ROOT / "results" / "tables.json").read_text(encoding="utf-8"))
R = T["runs"]
START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def p1(x: float) -> str:
    return str(Decimal(str(round(x, 6))).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


order = [("real_steady", "Steady"), ("real_burst", "Burst"), ("steady", "Steady"), ("burst", "Burst"),
         ("overload", "Sustained overload"), ("rpm_halved", "Per-minute limit halved"),
         ("daily_exhausted", "Daily budget exhausted")]
rn, rx, rf = (json.loads((ROOT / "results" / "runs" / k / "run_meta.json").read_text(encoding="utf-8"))["provider_stats"]
              for k in ("retry_none", "retry_naive", "retry_relayflow"))
S = T["slowest"]["overload"]

md = [START, "", "## Results", "",
      "Full two-page report: [`report/final_report.pdf`](report/final_report.pdf). All numbers below are computed from "
      "`results/results.csv` / `results/event_logs.jsonl` by `benchmarks/analysis.py`. "
      "**Real** rows are Groq free tier (`openai/gpt-oss-120b`); **stand-in** rows come from the local simulator "
      "and are not Groq measurements.", "",
      "### Table 1. Limits found in the real service", "",
      "| Limit | Value you found | How you found it | Confidence |", "|---|---|---|---|",
      "| Requests per minute | 30 | Docs; never reached | Medium (docs only) |",
      "| Tokens per minute | 8,000, charged at admission as prompt + max_tokens | Docs; `x-ratelimit-limit-tokens`=8000; real 429s with remaining-tokens 817-1,779 | High |",
      "| Requests per day | 1,000, refilling ~86.4 s/request (inferred) | Docs; `limit-requests`=1000; `reset-requests` 1m26.4s | Value high; refill medium |",
      "| Tokens per day | 200,000 | Docs only; no header; stayed under it, not tested | Low (not measured) |",
      "| Requests in flight | Not documented; 6 simultaneous accepted | Real burst run | Low |",
      "| Returned when a limit is hit | HTTP 429 + JSON error + `retry-after` (s) only on 429; `x-ratelimit-*` on every reply | 4 real 429s (Retry-After 1, 2, 7, 10 s) | High |",
      "| Time until it recovers | Per-minute: the Retry-After (retries succeeded). Daily: not measured | Real burst run | Minute: medium; day: none |",
      "", "### Table 2a. Latency (ms, successful requests; n/m = sample too small for that percentile)", "",
      "| Scenario | Real or stand-in | Requests sent | Duration (s) | Offered rate (per min) | First token p50/p95/p99 (ms) | Total time p50/p95/p99 (ms) |",
      "|---|---|---|---|---|---|---|"]
for k, n in order:
    r = R[k]
    md.append(f"| {n} | {r['source']} | {r['requests']} | {r['duration_s']:.0f} | {r['offered_per_min']:.0f} | {r['ttft']} | {r['total']} |")
md += ["", "### Table 2b. Outcomes (disjoint buckets; cost is notional list price, the free tier bills $0)", "",
       "| Scenario | Succeeded % | Rate-limited % | Timed out or given up % | Retries per request | Peak queue length | Tokens per answer | Cost per 1,000 answers |",
       "|---|---|---|---|---|---|---|---|"]
for k, n in order:
    r = R[k]
    md.append(f"| {n} ({r['source']}) | {p1(r['ok_pct'])} | {p1(r['rate_limited_pct'])} | {p1(r['given_up_pct'])} | "
              f"{r['retries_per_request']:.2f} | {r['peak_queue']} | {r['tokens_per_answer']:.1f} | ${r['cost_per_1000']:.2f} |")
md += ["", "### Table 3. Retrieval settings (real Groq; 16 questions each; % of all questions asked; "
       "dagger = p95 from n<20)", "",
       "| Setting | Questions asked | Correct and cited % | Wrong % | Correctly refused % | Tokens per answer | First token p50/p95 (ms) | Total time p50/p95 (ms) |",
       "|---|---|---|---|---|---|---|---|"]
names = {"A": "A: 300 tok x3, no rerank", "B": "B: 600 tok x4, no rerank", "C": "C: 600 tok x6, rerank"}
for r in T["retrieval"]:
    md.append(f"| {names[r['cfg']]} | {r['n']} | {p1(r['correct_pct'])} | {p1(r['wrong_pct'])} | "
              f"{p1(r['refused_correctly_pct'])} | {r['tokens_per_answer']:.0f} | {r['ttft_p50']}/{r['ttft_p95']} | "
              f"{r['total_p50']}/{r['total_p95']} |")
md += ["", "**Choice: B.** Best measured quality per token (C costs ~28% more tokens for no gain). After the run, two "
       "verifier bugs on real model output were found and fixed; replaying the stored drafts offline "
       "(`results/verifier_replay.json`) lifts B and C to 62.5% correct and leaves A at 50.0%. "
       "11 answerable questions are too few to rank configurations reliably, and settings were tuned on the same questions.",
       "", "### Key findings", "",
       f"* **Where the time goes:** the slowest request (stand-in overload) took {S['total_ms']/1000:.1f} s: "
       f"{S['queue_wait_ms']/1000:.1f} s queue + {S['limiter_wait_ms']/1000:.1f} s rate-limit wait + "
       f"{S['provider_ms']/1000:.2f} s model. All of it was before the first token.",
       f"* **Retries can make things worse:** same overloaded provider, no retries sent {rn['arrivals']} calls "
       f"({p1(R['retry_none']['ok_pct'])}% ok); naive immediate retries sent {rx['arrivals']} "
       f"(+{(rx['arrivals']/rn['arrivals']-1)*100:.0f}%, {rx['rejected_rpm']} rejected, {p1(R['retry_naive']['ok_pct'])}% ok); "
       f"RelayFlow's policy sent {rf['arrivals']} ({p1(R['retry_relayflow']['ok_pct'])}% ok) but users waited "
       f"(p95 first token {R['retry_relayflow']['ttft'].split('/')[1]} ms). One run each, small effect on success.",
       "* **Daily budget gone (compressed day, 15:00 = t=180 s):** 75% served, 25% got a transparent message; "
       "the provider saw only 3 daily rejections for 80 requests.",
       "* **Real-model surprises, all fixed with tests:** citations as `【S1】`, `[S3†L1-L5]`, \"100 000\" thousands "
       "separators; ~1.5 tokens/word instead of 1.3.",
       "", "Charts: `results/charts/queue_length.png`, `results/charts/retry_comparison.png`.", "", END]
block = "\n".join(md)
readme = (ROOT / "README.md").read_text(encoding="utf-8")
if START in readme:
    a, b = readme.index(START), readme.index(END) + len(END)
    readme = readme[:a] + block + readme[b:]
else:
    marker = "## Honest limitations"
    readme = readme.replace(marker, block + "\n\n" + marker, 1)
(ROOT / "README.md").write_text(readme, encoding="utf-8")
print("README results section updated")
