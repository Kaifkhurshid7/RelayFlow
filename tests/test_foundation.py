"""Phase 1 smoke tests: config and entry point."""
from fastapi.testclient import TestClient

from app.config import LIBRARY, RETRIEVAL_CONFIGS, Settings
from app.main import app


def test_library_has_exact_versions() -> None:
    got = {(p.arxiv_id, p.version) for p in LIBRARY}
    assert got == {
        ("1706.03762", "v7"),
        ("2410.00037", "v1"),
        ("2507.07955", "v2"),
    }
    assert LIBRARY[0].url == "https://arxiv.org/pdf/1706.03762v7"


def test_retrieval_configs_match_assignment() -> None:
    assert (RETRIEVAL_CONFIGS["A"].chunk_tokens, RETRIEVAL_CONFIGS["A"].top_k) == (300, 3)
    assert RETRIEVAL_CONFIGS["C"].rerank is True


def test_api_key_is_secret_and_required(monkeypatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    s = Settings(_env_file=None)
    try:
        s.require_groq_key()
        raise AssertionError("expected RuntimeError")
    except RuntimeError:
        pass
    s2 = Settings(_env_file=None, groq_api_key="gsk_secret")
    assert "gsk_secret" not in repr(s2)
    assert s2.require_groq_key() == "gsk_secret"


def test_nested_env_override(monkeypatch) -> None:
    monkeypatch.setenv("SCHEDULER__RPM", "15")
    assert Settings(_env_file=None).scheduler.rpm == 15


def test_health_endpoint() -> None:
    r = TestClient(app).get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert len(r.json()["library"]) == 3
