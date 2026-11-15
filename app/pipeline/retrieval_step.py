"""The pipeline's retrieval stage as one function, shared by the pipeline and the offline evaluation
so the two can never drift apart."""
from __future__ import annotations

import math
from typing import Callable

from app.pipeline.decision_engine import COVERAGE_GATE
from app.pipeline.question_analyzer import Analysis
from app.retrieval.retrieval import Passage, Retriever

# (tool_name, fn, *args) -> result ; lets the pipeline time each tool call
ToolRunner = Callable[..., object]


def retrieve(
    retr: Retriever,
    a: Analysis,
    question: str,
    run_tool: ToolRunner | None = None,
    gate: float = COVERAGE_GATE,
) -> tuple[list[Passage], float, str, int]:
    """Return (passages, coverage, scope_used, k_full).

    Loop over search scopes: restrict to the detected papers first. Widen to the whole library only
    when the papers were inferred from topic words; if the user NAMED a paper, never widen, because
    another paper's vocabulary (BLEU for "Moshi's BLEU score") would fake coverage.
    Cross-paper questions retrieve per paper and interleave so every named paper is represented.
    """
    call = run_tool or (lambda name, fn, *args: fn(*args))
    k_full = math.ceil(retr.config.top_k * a.depth_multiplier)
    scopes: list[set[str] | None] = [None]
    if a.restrict_papers and a.papers:
        scopes = [set(a.papers)] if a.named_papers else [set(a.papers), None]
    best: list[Passage] = []
    best_cov, scope_used = 0.0, "all"
    for scope in scopes:
        if a.cross_paper and scope and len(scope) >= 2:
            got = call("search_library", retr.search_balanced, question, sorted(scope), k_full)
        else:
            got = call("search_library", retr.search, question, k_full, scope)
        cov = call("evidence_coverage", retr.coverage, question, got)
        if cov > best_cov or not best:
            best, best_cov, scope_used = got, cov, ("restricted" if scope else "all")
        if cov >= gate:
            break
    return best, best_cov, scope_used, k_full  # type: ignore[return-value]
