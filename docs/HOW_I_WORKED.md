# How I worked with the assistant

**Kaif Khurshid**

I built RelayFlow with Claude Code (Anthropic's coding assistant, running inside VS Code) as my pair
programmer. I wrote the brief and the rules, handed over the accounts, set the limits on how the work
was delivered, and pushed it from one phase to the next until the experiments were run and the report
was written. The assistant wrote most of the code and ran the experiments; I directed it. The full list
of what I typed is in `docs/conversation_history.md` (my keys are redacted there). This page is the
honest account of how it went.

## What I actually asked for, in order
1. I pasted the full RelayFlow project brief (the assignment, the tech stack, the rules, the eight phases)
   and asked to set up git, a `.gitignore` and a README first so I could push to GitHub.
2. I asked for the whole folder structure to be created too, so it could be pushed.
3. I said to start building the project, but **never to commit anything**. I wanted to look at the work and
   push it myself.
4. I gave my Groq and Gemini keys and said "use".
5. I asked to test whether the keys really worked.
6. I asked whether everything was done, then told it to do all the phases properly, and later to do it fast.

## Decisions I made
* **Groq with `openai/gpt-oss-120b`** as the real model, because the assignment asks for a real hosted
  model on the free tier, and my Groq account was the one to use.
* **Exact paper versions** (Transformer v7, Moshi v1, H-Net v2), checked from the arXiv stamp on page 1.
* **Never commit.** Everything stays uncommitted until I decide to push.
* **Keys only in `.env`**, which is git-ignored. I did hand over a second Groq key and three Gemini keys,
  but the project stayed on Groq as the assignment requires, so the Gemini keys and the second Groq key
  were never used. I did not want extra accounts used to dodge the free-tier limits.
* **Keep the real experiments small.** The free tier allows 200,000 tokens a day, so the real quality run
  used 16 questions and a token cap. The heavy load tests ran against the local stand-in instead, and
  every table says which rows are real and which are stand-in.

## Mistakes that came up while building, and how they were fixed
These were found by the test suite and by looking at raw outputs while the work was going on. Each one got a
regression test.

| # | What went wrong | How it showed up | Fix |
|---|---|---|---|
| 1 | I assumed a low search score meant "the papers don't answer this" | a test: the Nvidia-revenue question scored 7.2 because of "NVIDIA P100 GPUs" | refusal no longer trusts the score; coverage check + model + verification |
| 2 | Bold table cells were read as section headings; Moshi's sub-headings were missed | printing the section list for each paper | sequence checks and a rule for small-caps headings |
| 3 | An invisible backspace character silently broke a regex | scanning the code for control characters | removed; Moshi's appendix F now parses |
| 4 | `1m26.4s` in a Retry-After hint was read as 60 seconds | a unit test | proper duration parser |
| 5 | The stand-in gave back unused tokens too early | a rate-limit test failed | refund only when the request completes |
| 6 | The offline retrieval check copied the pipeline's logic and drifted from it | one question looked answerable when it wasn't | one shared retrieval function |
| 7 | The real model writes citations as `【S1】`, uses thin spaces and writes "100 000" | the first real calls; I printed the raw text | text normalisation, a prompt change, tests |
| 8 | I assumed 1.3 tokens per word; real prompts cost about 1.5 | compared Groq's token counts with word counts | one shared constant |
| 9 | The grader accepted a two-paper answer where one paper's citation didn't support its claim | reading the pilot answer for the BPE question | per-paper checks |

I did not find these myself; the tests and the real runs did, and I am listing them because the report should
show what actually went wrong, not a clean story.

## Things we decided not to do
* A Gemini fallback provider, the second Groq key, and an LLM acting as judge for grading. Answers are graded
  from the ground truth (key facts, which paper and page was cited, and whether the facts are in the cited
  text), never by the model's opinion of itself.

## The three prompts that mattered most
1. **The project brief.** It fixed the scope, the exact papers, the tables and the honesty rules ("never
   fabricate", "label simulated results").
2. **"never commit any changes strictly".** A hard rule about how the work is delivered.
3. **"do all the phases properly".** It turned a plan into a finished, measured system.

## What is mine
The assignment framing and all the rules; choosing Groq and `gpt-oss-120b` and providing the accounts; the
no-commit rule and the decision about when to push; deciding how much of the free-tier budget to spend; and
the final review of the report and the Vaani feedback before I submit. The code, the experiments and the first
draft of the report were produced by the assistant under that direction, and the report says so.
