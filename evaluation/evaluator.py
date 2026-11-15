"""Deterministic answer grading against evaluation/ground_truth.json.

Grades (the four values allowed in results.csv):
  correct            answered, key facts present (cross-paper questions: each paper's facts must be in the
                     passages cited FROM THAT PAPER), forbidden claims absent, a cited passage comes from
                     the expected paper(s) on the expected pages, AND every key fact is actually in the
                     cited passage text (a citation's presence alone is never accepted)
  wrong              answered, but any of the above fails (including answering an unanswerable question)
  refused_correctly  refused a question that must be refused (or a false-premise one where refusal is acceptable)
  refused_wrongly    refused a question the library can answer

Requests with no answer or refusal produced (capacity messages, clarifications, failures) are not graded
(empty ``answer_grade``): the system did not attempt to answer from evidence.
The model's own opinion of its answer is never used.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from app.config import ROOT_DIR

EVAL_DIR = Path(ROOT_DIR) / "evaluation"
REFUSAL_TYPES = ("refusal_no_evidence", "refusal_model", "refusal_unverified")


def load_questions() -> list[dict]:
    return json.loads((EVAL_DIR / "questions.json").read_text(encoding="utf-8"))


def load_ground_truth() -> dict[str, dict]:
    gt = json.loads((EVAL_DIR / "ground_truth.json").read_text(encoding="utf-8"))
    return {k: v for k, v in gt.items() if not k.startswith("_")}


def _answer_text(ev: dict) -> str:
    return ev.get("answer", "").split("\n\nSources:")[0]


def _group_matches(group: list[str], text: str) -> bool:
    return any(re.search(p, text, re.I) for p in group)


def _pages_overlap(pages: list[int], support_pages: list[int], slack: int = 1) -> bool:
    return pages[0] <= support_pages[1] + slack and pages[1] >= support_pages[0] - slack


def grade_event(ev: dict, gt: dict, chunk_text: dict[str, str]) -> tuple[str, list[str]]:
    """Return (grade or '' if not gradable, reasons)."""
    if ev.get("status") != "ok" or ev.get("answer_type") in ("capacity_message", "clarification", "none"):
        return "", ["not graded: no answer or refusal was produced from evidence"]
    refused = ev.get("answer_type") in REFUSAL_TYPES
    beh = gt["expected_behavior"]
    if beh == "refuse":
        return ("refused_correctly", ["refused as required"]) if refused else (
            "wrong", ["answered a question the library cannot answer"])
    if refused:
        if gt.get("refusal_acceptable"):
            return "refused_correctly", ["refusal acceptable for this false-premise question"]
        return "refused_wrongly", ["refused a question the library answers"]

    reasons: list[str] = []
    text = _answer_text(ev)
    groups = gt["key_facts"]
    for g in groups:
        if not _group_matches(g, text):
            reasons.append(f"missing key fact {g[:2]}")
    for pat in gt.get("forbidden", []):
        if re.search(pat, text, re.I):
            reasons.append(f"forbidden claim {pat!r}")
    cited = ev.get("citation_chunks") or []
    expected_papers = {s["paper"] for s in gt["support"]}
    cited_expected = [c for c in cited if c["paper_id"] in expected_papers]
    if not cited:
        reasons.append("no citations")
    else:
        need = gt.get("min_papers_cited", 1)
        if len({c["paper_id"] for c in cited_expected}) < need:
            reasons.append(f"cites {len({c['paper_id'] for c in cited_expected})} expected paper(s), need {need}")
        page_ok = any(
            _pages_overlap(c["pages"], s["pages"])
            for c in cited_expected for s in gt["support"] if s["paper"] == c["paper_id"]
        )
        if not page_ok:
            reasons.append("cited passages are not on the supporting pages")
        union = " ".join(chunk_text.get(c["chunk_id"], "") for c in cited)
        in_evidence = sum(_group_matches(g, union) for g in groups)
        if groups and in_evidence < len(groups):
            reasons.append("key facts are not in the cited passage text")
    for paper, pgroups in (gt.get("paper_facts") or {}).items():
        ptext = " ".join(chunk_text.get(c["chunk_id"], "") for c in cited if c["paper_id"] == paper)
        for grp in pgroups:
            if not _group_matches(grp, ptext):
                reasons.append(f"claims about '{paper}' are not supported by the passages cited from it {grp[:2]}")
    return ("correct", ["all checks passed"]) if not reasons else ("wrong", reasons)


def chunk_text_lookup() -> dict[str, str]:
    """chunk_id -> text for every configured chunk size (used to inspect cited evidence)."""
    from app.runtime import get_retrievers

    out: dict[str, str] = {}
    for r in get_retrievers().values():
        out.update({c.chunk_id: c.text for c in r.chunks})
    return out


def grade_events(events: list[dict]) -> dict[str, dict]:
    """Grade events that carry a ``question_id``. Returns {request_id: {grade, reasons}}."""
    gt, texts = load_ground_truth(), chunk_text_lookup()
    out: dict[str, dict] = {}
    for ev in events:
        qid = ev.get("question_id")
        if qid in gt:
            g, why = grade_event(ev, gt[qid], texts)
            out[ev["request_id"]] = {"grade": g, "reasons": why, "question_id": qid}
    return out
