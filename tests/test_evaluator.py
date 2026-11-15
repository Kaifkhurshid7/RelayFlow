"""Evaluator behaviour: a citation's presence is never enough."""
from evaluation.evaluator import grade_event, load_ground_truth, load_questions

GT = load_ground_truth()
CHUNKS = {
    "attention-600-0010": "In this work we employ h = 8 parallel attention layers, or heads. dk = dv = dmodel/h = 64.",
    "attention-600-0017": "We trained our models on one machine with 8 NVIDIA P100 GPUs.",
    "moshi-600-0001": "Moshi has a theoretical latency of 160ms, 200ms in practice.",
}


def ev(answer, chunks, answer_type="answer", status="ok"):
    cc = [{"chunk_id": cid, "paper_id": cid.split("-")[0], "pages": pg, "section": "x"} for cid, pg in chunks]
    return {"status": status, "answer_type": answer_type, "answer": answer, "citation_chunks": cc}


def test_dataset_is_well_formed() -> None:
    qs = load_questions()
    assert len(qs) >= 20 and len({q["id"] for q in qs}) == len(qs)
    assert set(GT) == {q["id"] for q in qs}
    cats = {q["category"] for q in qs}
    assert {"direct_factual", "technical", "cross_paper", "limitations", "experimental_methods",
            "confusing_terminology"} <= cats
    assert any(c.startswith("unanswerable") for c in cats) and any(c.startswith("misleading") for c in cats)
    for k, g in GT.items():
        assert g["expected_behavior"] in ("answer", "refuse", "correct_premise")
        assert g["answerable"] == (g["expected_behavior"] != "refuse")
        if g["answerable"]:
            assert g["support"] and g["key_facts"] and g["expected_answer"]
        else:
            assert g["refusal_criteria"]


def test_correct_answer_is_correct() -> None:
    e = ev("The base Transformer uses 8 parallel heads [S1], each with dimension 64.", [("attention-600-0010", [5, 5])])
    assert grade_event(e, GT["Q01"], CHUNKS)[0] == "correct"


def test_citation_to_wrong_paper_or_pages_is_wrong() -> None:
    e = ev("8 heads of dimension 64 [S1].", [("moshi-600-0001", [1, 1])])
    g, why = grade_event(e, GT["Q01"], CHUNKS)
    assert g == "wrong" and any("expected paper" in r for r in why)
    e = ev("8 heads of dimension 64 [S1].", [("attention-600-0010", [9, 9])])
    assert grade_event(e, GT["Q01"], CHUNKS)[0] == "wrong"


def test_right_words_but_facts_not_in_cited_text_is_wrong() -> None:
    """Looks cited and correct, but the cited chunk says nothing about heads."""
    e = ev("The Transformer uses 8 heads with dimension 64 [S1].", [("attention-600-0017", [5, 5])])
    g, why = grade_event(e, GT["Q01"], CHUNKS)
    assert g == "wrong" and any("not in the cited passage" in r for r in why)


def test_missing_fact_is_wrong() -> None:
    e = ev("The Transformer uses multi-head attention [S1].", [("attention-600-0010", [5, 5])])
    assert grade_event(e, GT["Q01"], CHUNKS)[0] == "wrong"


def test_wrong_number_is_wrong() -> None:
    e = ev("The Transformer uses 16 heads [S1].", [("attention-600-0010", [5, 5])])
    assert grade_event(e, GT["Q13"], CHUNKS)[0] == "wrong"   # Q13 is about Moshi; paper+fact both off


def test_refusals() -> None:
    r = ev("The provided research library does not contain sufficient information to answer this question.",
           [], "refusal_no_evidence")
    assert grade_event(r, GT["Q25"], CHUNKS)[0] == "refused_correctly"
    assert grade_event(r, GT["Q01"], CHUNKS)[0] == "refused_wrongly"
    assert grade_event(r, GT["Q27"], CHUNKS)[0] == "refused_correctly"   # acceptable for that false premise
    assert grade_event(r, GT["Q28"], CHUNKS)[0] == "refused_wrongly"     # library does refute this premise


def test_answering_unanswerable_is_wrong() -> None:
    e = ev("Nvidia's revenue was strong [S1].", [("attention-600-0017", [7, 7])])
    assert grade_event(e, GT["Q25"], CHUNKS)[0] == "wrong"
    e = ev("The capital is Paris.", [], "answer")
    assert grade_event(e, GT["Q30"], CHUNKS)[0] == "wrong"


def test_not_gradable_cases_get_empty_grade() -> None:
    assert grade_event(ev("x", [], "capacity_message", status="rate_limited"), GT["Q01"], CHUNKS)[0] == ""
    assert grade_event(ev("x", [], "clarification"), GT["Q01"], CHUNKS)[0] == ""


def test_cross_paper_claims_must_be_supported_by_each_papers_own_citation() -> None:
    """Regression from the pilot: the Transformer's cited passage never mentions BPE; the facts were only
    'found' because the H-Net half of the answer mentioned BPE."""
    chunks = {"attention-600-0003": "Each token is converted to a learned embedding vector of dimension dmodel.",
              "attention-600-0020": "Sentences were encoded using byte-pair encoding with a shared vocabulary.",
              "dynamic_chunking-600-0001": "H-Net models raw bytes end to end without tokenization, replacing BPE."}
    ans = "The Transformer assumes pre-tokenized input [S1]. H-Net works on raw bytes instead of BPE tokens [S2]."
    bad = ev(ans, [("attention-600-0003", [5, 6]), ("dynamic_chunking-600-0001", [1, 1])])
    g, why = grade_event(bad, GT["Q22"], chunks)
    assert g == "wrong" and any("'attention'" in r for r in why)
    good_ans = "The Transformer uses byte-pair encoding [S1]. H-Net models raw bytes end to end without tokenization [S2]."
    good = ev(good_ans, [("attention-600-0020", [7, 7]), ("dynamic_chunking-600-0001", [1, 1])])
    assert grade_event(good, GT["Q22"], chunks)[0] == "correct"
