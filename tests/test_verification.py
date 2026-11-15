"""Citation / evidence verification tests (no network, no PDFs)."""
import pytest

from app.pipeline.prompts import REFUSAL
from app.retrieval.paper_ingestion import Chunk
from app.retrieval.retrieval import Passage
from app.verification.verification import verify_answer

HEADS = (
    "In this work we employ h = 8 parallel attention layers, or heads. For each of these we use "
    "dk = dv = dmodel/h = 64. Due to the reduced dimension of each head, the total computational cost "
    "is similar to that of single-head attention with full dimensionality."
)
HW = "We trained our models on one machine with 8 NVIDIA P100 GPUs. The base models were trained for 100,000 steps."


def passage(text: str, n: int = 0, section: str = "3.2.2 Multi-Head Attention", page: int = 4) -> Passage:
    c = Chunk(f"attention-600-{n:04d}", "attention", "Attention Is All You Need", "1706.03762", "v7",
              page, page, section, text, len(text.split()))
    return Passage(c, 5.0, n + 1)


P = [passage(HEADS), passage(HW, 1, "5.2 Hardware and Schedule", 7)]
Q = "How many parallel attention heads does the Transformer use?"


def test_valid_answer_passes_and_citation_is_expanded() -> None:
    r = verify_answer("The Transformer employs h = 8 parallel attention heads [S1].", Q, P)
    assert r.passed and not r.is_refusal
    c = r.citations[0]
    assert (c.arxiv_id, c.version, c.page_start, c.section) == ("1706.03762", "v7", 4, "3.2.2 Multi-Head Attention")
    assert "arXiv:1706.03762v7" in c.render()


def test_tag_after_period_is_normalised() -> None:
    assert verify_answer("The Transformer employs h = 8 parallel attention heads. [S1]", Q, P).passed


def test_number_word_matches_digit() -> None:
    assert verify_answer("The Transformer uses eight parallel attention heads [S1].", Q, P).passed


def test_fabricated_citation_fails() -> None:
    r = verify_answer("The Transformer employs 8 parallel attention heads [S5].", Q, P)
    assert not r.passed and any(i.startswith("citation_not_found") for i in r.issues)


def test_missing_citation_fails() -> None:
    r = verify_answer("The Transformer employs 8 parallel attention heads.", Q, P)
    assert not r.passed and r.issues[0].startswith("no_citation")


def test_wrong_number_fails_even_with_valid_citation() -> None:
    r = verify_answer("The Transformer employs 16 parallel attention heads [S1].", Q, P)
    assert not r.passed and any(i.startswith("unsupported_number") for i in r.issues)


def test_external_knowledge_fails() -> None:
    ans = "Attention heads were first popularised by BERT, which revolutionised natural language understanding [S1]."
    r = verify_answer(ans, Q, P)
    assert not r.passed and any(i.startswith("unsupported_claim") for i in r.issues)


def test_irrelevant_citation_fails() -> None:
    r = verify_answer("The models were trained on 8 NVIDIA P100 GPUs [S2].", Q, P)
    assert not r.passed and any(i.startswith("irrelevant_citation") for i in r.issues)


def test_wrong_version_and_unknown_paper_fail() -> None:
    r = verify_answer("Per arXiv:1706.03762v5 the model employs h = 8 parallel attention heads [S1].", Q, P)
    assert any(i.startswith("wrong_version") for i in r.issues)
    r = verify_answer("See arXiv:2001.00001 where h = 8 parallel attention heads are used [S1].", Q, P)
    assert any(i.startswith("unknown_paper") for i in r.issues)


def test_refusal_is_valid_but_cannot_cite() -> None:
    assert verify_answer(REFUSAL, "quarterly revenue of Nvidia?", P).is_refusal
    assert verify_answer(REFUSAL, "x", P).passed
    assert not verify_answer(REFUSAL + " [S1]", "x", P).passed


def test_misleading_relevant_looking_passage_is_not_proof() -> None:
    """A passage that mentions NVIDIA exists, but cannot support a revenue claim."""
    q = "What was Nvidia's quarterly revenue?"
    r = verify_answer("Nvidia reported quarterly revenue growth driven by GPU demand [S2].", q, P)
    assert not r.passed


def test_named_paper_must_be_cited_catches_entity_mixup() -> None:
    """'Moshi's BLEU' answered from the Transformer paper: the number IS in the cited text,
    so lexical support passes - only the named-paper check can catch it."""
    bleu = passage("Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task.", 3,
                   "6.1 Machine Translation", 8)
    q = "What BLEU score does Moshi achieve on the WMT 2014 English-to-German translation task?"
    ans = "Moshi achieves 28.4 BLEU on the WMT 2014 English-to-German translation task [S1]."
    assert verify_answer(ans, q, [bleu]).passed                                  # lexically fine...
    r = verify_answer(ans, q, [bleu], required_papers={"moshi"})                 # ...but Moshi is not cited
    assert not r.passed and any(i.startswith("named_paper_not_cited") for i in r.issues)


def test_cross_paper_answer_must_cite_every_named_paper() -> None:
    other = passage("The model removes tokenization and learns chunk boundaries end to end.", 5, "Abstract", 1)
    other.chunk.paper_id = "dynamic_chunking"
    q = "How do the Transformer paper and the H-Net paper treat tokenization?"
    one = verify_answer("The Transformer uses byte-pair encoding for tokenization [S1].", q,
                        [passage("Sentences were encoded using byte-pair encoding, a shared vocabulary.", 4), other],
                        required_papers={"attention", "dynamic_chunking"})
    assert any("dynamic_chunking" in i for i in one.issues)


def test_real_model_output_quirks_are_normalised() -> None:
    """Regression: gpt-oss-120b wrote \u3010S1\u3011 with U+202F / U+2011 characters; verification saw no citation."""
    from app.verification.verification import normalize_model_text

    raw = "The base Transformer employs eight parallel attention heads, each of dimension 64 (dk\u202f=\u202fdv\u202f=\u202f64)\u3010S1\u3011."
    norm = normalize_model_text(raw)
    assert "[S1]" in norm and "\u3010" not in norm and "\u202f" not in norm
    r = verify_answer(norm, Q, P)
    assert r.passed and r.citations[0].tag == "S1"
    assert normalize_model_text("key\u2011value") == "key-value"


def test_thousands_separators_and_line_ref_citations_from_real_model_output() -> None:
    """Regressions from the real run: '100 000' vs '100,000', and '[S1\u2020L1-L5]' citation tags."""
    from app.textutil import numbers
    from app.verification.verification import normalize_model_text

    assert numbers("trained for 100 000 steps") == numbers("trained for 100,000 steps") == {"100000"}
    assert numbers("batch of 25000 tokens and 3.5 days") == {"25000", "3.5"}
    src = passage(HW)
    ans = "The base models were trained on 8 NVIDIA P100 GPUs for 100 000 steps [S1\u2020L1-L5]."
    assert "[S1]" in normalize_model_text(ans)
    assert verify_answer(normalize_model_text(ans), "How were the base models trained, on which GPUs?", [src]).passed
