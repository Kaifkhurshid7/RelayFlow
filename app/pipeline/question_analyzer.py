"""Question understanding: topic, relevant papers, complexity, retrieval depth,
and whether a clarification is needed.

Rules run first (free, instant, deterministic). An optional LLM call is made only
when the rules find no paper at all and capacity is available; if that call fails
for any reason the rule result is used unchanged (the lightweight fallback).
Answerability is NOT decided here: only the retrieved evidence can establish it
(see ``Retriever.coverage`` and verification).
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

from app.retrieval.retrieval import tokenize

# Keywords that identify a paper. "strong" ones are specific enough to restrict
# retrieval to that paper; weak ones only bias the choice.
# Explicit names of a paper or of an artefact it introduces. If one of these appears, the user is
# asking about that paper, and an answer must cite it (see verification.required_papers).
NAMED_KEYWORDS: dict[str, list[str]] = {
    "attention": ["attention is all you need", "transformer paper", "original transformer", "vaswani"],
    "moshi": ["moshi", "helium", "mimi"],
    "dynamic_chunking": ["h-net", "hnet", "h net", "dynamic chunking"],
}
# Topic keywords. "strong" ones are specific to one paper; "weak" ones only bias the choice.
# "transformer" is deliberately weak: Moshi and H-Net also build on Transformers.
PAPER_KEYWORDS: dict[str, dict[str, list[str]]] = {
    "attention": {
        "strong": ["scaled dot-product", "multi-head attention", "positional encoding", "wmt", "bleu",
                   "label smoothing"],
        "weak": ["transformer", "self-attention", "encoder", "decoder", "translation", "attention head"],
    },
    "moshi": {
        "strong": ["inner monologue", "full-duplex", "rq-transformer", "spoken dialogue", "split rvq",
                   "depth transformer"],
        "weak": ["speech", "audio", "codec", "spoken", "dialogue", "streaming", "latency"],
    },
    "dynamic_chunking": {
        "strong": ["dechunking", "routing module", "ratio loss", "tokenizer-free", "byte-level",
                   "smoothing module"],
        "weak": ["chunking", "hierarchical", "byte", "tokenization", "tokenizer", "downsampler", "upsampler"],
    },
}
COMPARE_CUES = re.compile(r"\b(compare|compared|comparison|differ|differs|difference|differences|versus|vs\.?|"
                          r"both|in common|similar|similarities|contrast|each of the papers|all three|across)\b", re.I)
COMPLEX_CUES = re.compile(r"\b(why|explain|how does|how do|trade-?off|limitation|limitations|ablation|"
                          r"compare|differ|derive|reason|motivat\w+|justify|advantages?|disadvantages?)\b", re.I)
VAGUE = re.compile(r"^\s*(what|how|why)\s+(about|of)\s+(it|that|this|them)\b|^\s*(explain|describe|tell me about)\s+(it|that|this)\s*\??\s*$", re.I)


@dataclass
class Analysis:
    question: str
    topic: str
    papers: list[str] = field(default_factory=list)
    named_papers: list[str] = field(default_factory=list)  # explicitly named by the user
    restrict_papers: bool = False       # strong evidence -> restrict retrieval to ``papers``
    complexity: str = "simple"          # simple | moderate | complex
    multi_passage: bool = False
    cross_paper: bool = False
    depth_multiplier: float = 1.0       # scales the configured top_k
    needs_clarification: bool = False
    clarification_reason: str | None = None
    method: str = "rules"               # rules | llm+rules
    duration_ms: float = 0.0


def analyze_rules(question: str) -> Analysis:
    """Deterministic analysis; the always-available fallback."""
    t0 = time.perf_counter()
    q = question.strip()
    low = q.lower()
    toks = tokenize(q)
    a = Analysis(question=q, topic=" ".join(toks[:4]) or q[:40])

    named = [pid for pid, kws in NAMED_KEYWORDS.items() if any(k in low for k in kws)]
    strong, weak = [], []
    for pid, kw in PAPER_KEYWORDS.items():
        if any(k in low for k in kw["strong"]):
            strong.append(pid)
        elif any(k in low for k in kw["weak"]):
            weak.append(pid)
    a.named_papers = named
    # A named paper wins outright: terms from other papers (e.g. BLEU in "Moshi's BLEU score")
    # must not pull the search toward them.
    a.papers = named or strong or weak
    a.restrict_papers = bool(named or strong)

    a.cross_paper = len(named) >= 2 or (bool(COMPARE_CUES.search(low)) and len(a.papers) >= 2) \
        or (not named and len(strong) >= 2)
    n_clauses = len(re.findall(r"\b(and|or|while|whereas|but)\b|[;,]", low))
    score = (1 if COMPLEX_CUES.search(low) else 0) + (1 if len(toks) > 12 else 0) + (1 if n_clauses >= 2 else 0) \
        + (1 if a.cross_paper else 0)
    a.complexity = "simple" if score == 0 else "moderate" if score == 1 else "complex"
    a.multi_passage = a.cross_paper or a.complexity != "simple"
    a.depth_multiplier = {"simple": 1.0, "moderate": 1.0, "complex": 1.5}[a.complexity]
    if COMPARE_CUES.search(low) and not a.papers:
        a.papers, a.cross_paper = list(PAPER_KEYWORDS), True  # comparison with no named papers: search everywhere
    if a.cross_paper and len(a.papers) >= 2:
        a.restrict_papers = True

    if len(toks) < 2 or VAGUE.search(q):
        a.needs_clarification = True
        a.clarification_reason = "The question is too short or refers to something I cannot identify."
    a.duration_ms = (time.perf_counter() - t0) * 1000
    return a


LLM_SYSTEM = (
    "Classify a question about three research papers: attention (Attention Is All You Need), "
    "moshi (Moshi speech-text dialogue model), dynamic_chunking (H-Net dynamic chunking). "
    'Reply with JSON only: {"papers": [ids that the question is about, possibly empty], '
    '"complexity": "simple|moderate|complex", "topic": "<=5 words"}'
)


def parse_llm_analysis(text: str, base: Analysis) -> Analysis | None:
    """Merge a model's JSON classification into ``base``; None if unusable."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    papers = [p for p in d.get("papers", []) if p in PAPER_KEYWORDS]
    if d.get("complexity") in ("simple", "moderate", "complex"):
        base.complexity = d["complexity"]
        base.multi_passage = base.multi_passage or base.complexity != "simple"
    if papers:
        base.papers = papers
        base.restrict_papers = False  # model-suggested papers bias, never restrict
    if isinstance(d.get("topic"), str) and d["topic"].strip():
        base.topic = d["topic"].strip()[:60]
    base.method = "llm+rules"
    return base
