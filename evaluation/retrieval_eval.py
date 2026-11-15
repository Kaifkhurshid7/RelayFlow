"""Offline retrieval evaluation (no LLM, no API calls).

For each retrieval configuration and each question, run the same analysis +
retrieval + coverage-gate logic as the pipeline and measure:
  * source_hit   : a retrieved passage is from an expected paper on the supporting pages (+/-1)
  * fact_recall  : share of key-fact groups found in the union of retrieved text
  * all_facts    : every key-fact group is present in the retrieved text (upper bound on answer quality)
  * refuses      : the coverage gate would refuse (no API call)
  * est tokens   : estimated prompt tokens for the passages kept

Usage:  python -m evaluation.retrieval_eval [--gate 0.5]
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

from app.config import RETRIEVAL_CONFIGS, ROOT_DIR
from app.pipeline.decision_engine import COVERAGE_GATE
from app.pipeline.question_analyzer import analyze_rules
from app.pipeline.retrieval_step import retrieve
from app.runtime import get_retrievers
from evaluation.evaluator import _group_matches, _pages_overlap, load_ground_truth, load_questions


def retrieve_like_pipeline(retr, question: str):
    """The pipeline's own retrieval stage (shared code, so evaluation cannot drift from production)."""
    a = analyze_rules(question)
    passages, cov, _scope, _k = retrieve(retr, a, question)
    return a, passages, cov


def run(gate: float = COVERAGE_GATE, verbose: bool = False) -> dict:
    qs, gt, retrs = load_questions(), load_ground_truth(), get_retrievers()
    out: dict[str, dict] = {}
    for name in RETRIEVAL_CONFIGS:
        retr = retrs[name]
        rows = []
        for q in qs:
            g = gt[q["id"]]
            a, passages, cov = retrieve_like_pipeline(retr, q["question"])
            text = " ".join(p.chunk.text + " " + p.chunk.section for p in passages)
            row = {"id": q["id"], "answerable": g["answerable"], "coverage": round(cov, 3),
                   "refuses": cov < gate or not passages, "k": len(passages),
                   "est_tokens": sum(p.chunk.n_tokens_est for p in passages)}
            if g["answerable"]:
                exp = {s["paper"]: s["pages"] for s in g["support"]}
                row["source_hit"] = any(
                    p.chunk.paper_id in exp and _pages_overlap([p.chunk.page_start, p.chunk.page_end], exp[p.chunk.paper_id])
                    for p in passages)
                hits = [_group_matches(grp, text) for grp in g["key_facts"]]
                row["fact_recall"] = sum(hits) / len(hits)
                row["all_facts"] = all(hits)
            rows.append(row)
            if verbose:
                print(name, row)
        out[name] = {"rows": rows}
        ans = [r for r in rows if r["answerable"]]
        unans = [r for r in rows if not r["answerable"]]
        out[name]["summary"] = {
            "questions": len(rows), "answerable": len(ans),
            "source_hit_pct": 100 * sum(r["source_hit"] for r in ans) / len(ans),
            "fact_recall_pct": 100 * sum(r["fact_recall"] for r in ans) / len(ans),
            "all_facts_pct": 100 * sum(r["all_facts"] for r in ans) / len(ans),
            "gate_refuses_unanswerable_pct": 100 * sum(r["refuses"] for r in unans) / len(unans),
            "gate_wrongly_refuses_answerable_pct": 100 * sum(r["refuses"] for r in ans) / len(ans),
            "mean_est_prompt_tokens": sum(r["est_tokens"] for r in rows) / len(rows),
        }
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", type=float, default=COVERAGE_GATE)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--save", action="store_true")
    ns = ap.parse_args()
    res = run(ns.gate, ns.verbose)
    for name, d in res.items():
        s = d["summary"]
        print(f"{name}: " + "  ".join(f"{k}={v:.1f}" if isinstance(v, float) else f"{k}={v}" for k, v in s.items()))
    if ns.save:
        p = Path(ROOT_DIR) / "results" / "retrieval_eval.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"gate": ns.gate, "configs": res}, indent=1), encoding="utf-8")
        print("saved", p)
