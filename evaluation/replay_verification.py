"""Offline replay: re-verify the stored REAL model drafts with the current verifier.

Why: two verifier bugs were found on real gpt-oss-120b output after the quality run (thousands
separators like "100 000"; citation tags like "[S3†L1-L5]"). The measured events are never edited;
this script shows what the grades would have been with the fixed verifier, using exactly the model
text that was generated. Latency/tokens of the replayed requests are NOT recomputed (the real run
spent a repair call on them that the fixed verifier would not have needed).

Output: results/verifier_replay.json
"""
from __future__ import annotations

import json
from pathlib import Path

from app.config import ROOT_DIR
from app.monitoring.metrics import load_events
from app.pipeline.prompts import REFUSAL
from app.retrieval.retrieval import Passage
from app.runtime import get_retrievers
from app.verification.verification import normalize_model_text, verify_answer
from evaluation.evaluator import chunk_text_lookup, grade_event, load_ground_truth


def replay(run: str = "quality_real") -> dict:
    run_dir = Path(ROOT_DIR) / "results" / "runs" / run
    events = load_events(run_dir / "events.jsonl")
    grades = json.loads((run_dir / "grades.json").read_text(encoding="utf-8"))
    gt, texts = load_ground_truth(), chunk_text_lookup()
    chunks = {c.chunk_id: c for r in get_retrievers().values() for c in r.chunks}
    out = {"run": run, "changed": [], "unchanged_n": 0}
    for e in events:
        if not e.get("generations") or e["answer_type"] != "refusal_unverified":
            out["unchanged_n"] += 1
            continue
        plan = e["plan"]
        used = [Passage(chunks[p["chunk_id"]], p["score"], i)
                for i, p in enumerate(plan["passages"][: plan["top_k"]], 1)]
        named = set(e["analysis"].get("named_papers", []))
        new = None
        for g in e["generations"]:
            text = normalize_model_text(g["text"]).strip()
            v = verify_answer(text, e["question"], used, named)
            if v.passed:
                new = (text, v)
                break
        old_grade = grades[e["request_id"]]["grade"]
        if new is None:
            out["unchanged_n"] += 1
            continue
        text, v = new
        e2 = dict(e)
        if v.is_refusal:
            e2.update(answer=REFUSAL, answer_type="refusal_model", citation_chunks=[])
        else:
            e2.update(answer=text + "\n\nSources:", answer_type="answer",
                      citation_chunks=[{"chunk_id": c.chunk_id, "paper_id": c.paper_id,
                                        "pages": [c.page_start, c.page_end], "section": c.section}
                                       for c in v.citations])
        g2, why = grade_event(e2, gt[e["question_id"]], texts)
        out["changed"].append({"request_id": e["request_id"], "question_id": e["question_id"],
                               "config": e["retrieval_setting"], "old_grade": old_grade, "new_grade": g2,
                               "reasons": why})
    out["summary"] = {k: sum(1 for c in out["changed"] if c["new_grade"] == k)
                      for k in ("correct", "wrong", "refused_correctly", "refused_wrongly")}
    return out


if __name__ == "__main__":
    res = replay()
    p = Path(ROOT_DIR) / "results" / "verifier_replay.json"
    p.write_text(json.dumps(res, indent=1), encoding="utf-8")
    for c in res["changed"]:
        print(c["question_id"], c["config"], c["old_grade"], "->", c["new_grade"], c["reasons"][:1])
    print("wrote", p)
