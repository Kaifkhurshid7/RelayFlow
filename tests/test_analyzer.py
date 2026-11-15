"""Question analysis rules."""
import pytest

from app.pipeline.question_analyzer import analyze_rules, parse_llm_analysis


@pytest.mark.parametrize("q,named,cross", [
    ("What is the inner monologue in Moshi?", ["moshi"], False),
    ("How do the Transformer paper and the H-Net paper differ in tokenization?", ["attention", "dynamic_chunking"], True),
    ("What BLEU score does Moshi achieve on WMT 2014?", ["moshi"], False),   # BLEU must not pull in the Attention paper
    ("How many attention heads does the Depth Transformer in Moshi use?", ["moshi"], False),
])
def test_named_papers_and_cross_paper(q, named, cross) -> None:
    a = analyze_rules(q)
    assert a.named_papers == named and a.cross_paper is cross
    assert a.papers == named and a.restrict_papers


def test_topic_words_infer_but_do_not_name_a_paper() -> None:
    a = analyze_rules("What positional encoding is used and why?")
    assert a.papers == ["attention"] and a.named_papers == []


def test_unrelated_question_has_no_paper() -> None:
    a = analyze_rules("What is the capital of France?")
    assert a.papers == [] and not a.needs_clarification


@pytest.mark.parametrize("q", ["What about it?", "why?", "Explain it"])
def test_vague_questions_need_clarification(q) -> None:
    assert analyze_rules(q).needs_clarification


def test_complexity_scales_depth() -> None:
    assert analyze_rules("What is the ratio loss?").complexity == "simple"
    c = analyze_rules("Compare and explain why Moshi and H-Net differ in how they reduce sequence length")
    assert c.complexity == "complex" and c.depth_multiplier > 1 and c.cross_paper


def test_llm_assist_merges_json_and_never_restricts() -> None:
    base = analyze_rules("Tell me about residual quantization codebooks")
    out = parse_llm_analysis('Sure: {"papers": ["moshi"], "complexity": "moderate", "topic": "audio codec"}', base)
    assert out is not None and out.papers == ["moshi"] and out.method == "llm+rules" and not out.restrict_papers
    assert parse_llm_analysis("not json at all", analyze_rules("x y z")) is None
    assert parse_llm_analysis('{"papers": ["bogus"]}', analyze_rules("x y z")).papers == []   # unknown ids ignored
