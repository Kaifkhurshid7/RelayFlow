"""Ingestion and retrieval tests (require the downloaded PDFs)."""
import pytest

from app.config import LIBRARY, RETRIEVAL_CONFIGS, get_settings
from app.retrieval.paper_ingestion import build_chunks, est_tokens
from app.retrieval.retrieval import Retriever, tokenize

pytestmark = pytest.mark.skipif(
    not all((get_settings().papers_dir / p.filename).exists() for p in LIBRARY),
    reason="run `python -m app.retrieval.download` first",
)


@pytest.fixture(scope="module")
def retr_b() -> Retriever:
    return Retriever.from_config("B")


def test_chunks_have_metadata_and_size_bound() -> None:
    chunks = build_chunks(300)
    assert {c.paper_id for c in chunks} == {p.paper_id for p in LIBRARY}
    for c in chunks:
        assert c.page_start >= 1 and c.page_end >= c.page_start
        assert c.section and c.version in {"v7", "v1", "v2"}
        assert c.n_tokens_est <= 300 + 5  # word-rounding tolerance


def test_references_are_not_indexed() -> None:
    assert not any(c.section.lower().startswith("references") for c in build_chunks(600))


def test_stemming_matches_inflections() -> None:
    assert tokenize("practical latency heads") == tokenize("practice latencies head")


def test_token_estimate() -> None:
    from app.textutil import TOKENS_PER_WORD

    assert est_tokens("one two three four") == round(4 * TOKENS_PER_WORD)


def test_tokenize_drops_stopwords() -> None:
    assert tokenize("What is the Scaled Dot-Product Attention?") == ["scale", "dot-product", "attent"]


@pytest.mark.parametrize(
    "question,paper_id,section_prefix",
    [
        ("How many attention heads does the base Transformer use?", "attention", "3.2"),
        ("What is the inner monologue in Moshi?", "moshi", "3.4"),
        ("What is the ratio loss in H-Net dynamic chunking?", "dynamic_chunking", "2.2"),
    ],
)
def test_known_questions_hit_right_paper_and_section(retr_b, question, paper_id, section_prefix) -> None:
    hits = retr_b.search(question)
    assert hits and hits[0].chunk.paper_id == paper_id
    assert any(h.chunk.section.startswith(section_prefix) for h in hits)


def test_possessives_are_stripped() -> None:
    assert tokenize("The Transformer’s encoder and Nvidia's GPUs") == ["transform", "encod", "nvidia", "gpus"]


def test_paper_filter(retr_b) -> None:
    hits = retr_b.search("attention", paper_ids={"moshi"})
    assert hits and all(h.chunk.paper_id == "moshi" for h in hits)


def test_rerank_config_returns_top_k() -> None:
    r = Retriever.from_config("C")
    assert len(r.search("multi-head attention heads dimension")) <= RETRIEVAL_CONFIGS["C"].top_k


def test_out_of_library_query_still_returns_lexical_hit(retr_b) -> None:
    """Regression for a real failure mode: BM25 matches a rare token ("NVIDIA" in
    the hardware paragraph) for a question the library cannot answer. Scores are
    therefore NOT a safe refusal signal; verification must decide."""
    hits = retr_b.search("quarterly revenue of Nvidia")
    assert hits, "expected a lexical hit on the rare token"
    top = hits[0].chunk.text.lower()
    assert "nvidia" in top and "revenue" not in top
