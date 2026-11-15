"""Citation and evidence verification (deterministic, no LLM).

Design: the model never writes citations itself. It emits ``[S#]`` tags that map
to the passages we retrieved; we expand those tags into full citations from the
passage metadata (paper, arXiv id, exact version, section, page). A tag that does
not map to a retrieved passage is a fabricated citation and fails verification.
A citation's mere presence is not accepted as proof: each answer sentence must
also be *supported* by the text of the passages cited, including every number.
"""
from __future__ import annotations

import re
import time
import unicodedata
from dataclasses import dataclass, field

from app.config import LIBRARY
from app.pipeline.prompts import REFUSAL
from app.retrieval.retrieval import Passage
from app.textutil import content_stems, numbers

TAG_GROUP = re.compile(r"\[\s*S\d+(?:\s*[,;]\s*S\d+)*\s*\]")
_TAG_SUFFIX = re.compile(r"(\[\s*S\d+)\s*\u2020[^\]\[]*\]")  # [S3\u2020L1-L5] -> [S3]
TAG_NUM = re.compile(r"S(\d+)")
ARXIV_MENTION = re.compile(r"arXiv:?\s*(\d{4}\.\d{5})(v\d+)?", re.I)

SUPPORT_THRESHOLD = 0.5   # fraction of a sentence's content stems found in cited text
RELEVANCE_THRESHOLD = 0.3  # fraction of question stems found in the cited text


@dataclass
class Citation:
    tag: str
    paper_id: str
    paper_title: str
    arxiv_id: str
    version: str
    section: str
    page_start: int
    page_end: int
    chunk_id: str

    def render(self) -> str:
        pg = f"p.{self.page_start}" if self.page_start == self.page_end else f"pp.{self.page_start}-{self.page_end}"
        return f"[{self.tag}] {self.paper_title} (arXiv:{self.arxiv_id}{self.version}), §{self.section}, {pg}"


@dataclass
class VerificationResult:
    passed: bool
    is_refusal: bool
    citations: list[Citation] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    min_support: float | None = None
    duration_ms: float = 0.0

    def feedback(self) -> str:
        return "; ".join(self.issues)


_BRACKETS = str.maketrans({"\u3010": "[", "\u3011": "]", "\uff3b": "[", "\uff3d": "]", "\u27e6": "[", "\u27e7": "]",
                           "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2212": "-"})


def normalize_model_text(text: str) -> str:
    """Make model output machine-checkable.

    Observed on openai/gpt-oss-120b: citations come back as \u3010S1\u3011 (lenticular brackets) instead
    of [S1], with narrow no-break spaces (U+202F) and non-breaking hyphens (U+2011) in the prose. NFKC
    plus a bracket/hyphen map restores plain ASCII; wording is untouched.
    """
    text = unicodedata.normalize("NFKC", text).translate(_BRACKETS)
    return _TAG_SUFFIX.sub(r"\1]", text)


def is_refusal(text: str) -> bool:
    return REFUSAL.lower() in text.strip().lower()


def _sentences(text: str) -> list[str]:
    # move tags that follow a full stop back before it, so "x. [S1] y" -> "x [S1]. y"
    text = re.sub(r"([.!?])\s*(" + TAG_GROUP.pattern + r"(?:\s*" + TAG_GROUP.pattern + r")*)", r" \2\1", text)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])", text.strip())
    return [p.strip() for p in parts if p.strip()]


def verify_answer(
    answer: str,
    question: str,
    passages: list[Passage],
    required_papers: set[str] | None = None,
) -> VerificationResult:
    """Verify ``answer`` against the retrieved ``passages`` it may cite.

    ``required_papers``: papers the question names explicitly; a non-refusal answer must cite each of
    them. This catches entity mix-ups that lexical support cannot, e.g. attributing the Transformer's
    BLEU score to Moshi: the number IS in the cited text, but the named paper is not cited.
    """
    t0 = time.perf_counter()
    res = VerificationResult(passed=False, is_refusal=False)

    def done() -> VerificationResult:
        res.passed = not res.issues
        res.duration_ms = (time.perf_counter() - t0) * 1000
        return res

    if is_refusal(answer):
        res.is_refusal = True
        if TAG_GROUP.search(answer):
            res.issues.append("refusal_with_citations: a refusal must not cite sources")
        return done()

    # 1. citation tags must exist and map to retrieved passages
    tags = [int(n) for g in TAG_GROUP.findall(answer) for n in TAG_NUM.findall(g)]
    if not tags:
        res.issues.append("no_citation: answer has no [S#] citations")
        return done()
    seen: dict[int, Citation] = {}
    for n in tags:
        if not 1 <= n <= len(passages):
            res.issues.append(f"citation_not_found: [S{n}] does not match any retrieved passage")
        elif n not in seen:
            c = passages[n - 1].chunk
            seen[n] = Citation(f"S{n}", c.paper_id, c.paper_title, c.arxiv_id, c.version, c.section,
                               c.page_start, c.page_end, c.chunk_id)
    res.citations = list(seen.values())

    # 2. explicit arXiv mentions must be library papers at the exact version
    lib = {p.arxiv_id: p.version for p in LIBRARY}
    for aid, ver in ARXIV_MENTION.findall(answer):
        if aid not in lib:
            res.issues.append(f"unknown_paper: arXiv:{aid} is not in the library")
        elif ver and ver != lib[aid]:
            res.issues.append(f"wrong_version: arXiv:{aid}{ver} but the library uses {lib[aid]}")

    for pid in sorted(required_papers or ()):
        if not any(c.paper_id == pid for c in res.citations):
            res.issues.append(f"named_paper_not_cited: the question names '{pid}' but no cited passage is from it")

    cited_idx = [n for n in seen]
    if not cited_idx:
        return done()
    all_cited_text = " ".join(passages[n - 1].chunk.text for n in cited_idx)

    # 3. cited passages must be relevant to the question
    q_stems = content_stems(question)
    if q_stems:
        rel = len(q_stems & content_stems(all_cited_text)) / len(q_stems)
        if rel < RELEVANCE_THRESHOLD:
            res.issues.append(f"irrelevant_citation: cited passages cover only {rel:.0%} of the question terms")

    # 4. every substantive sentence must be supported by the passages it cites
    supports: list[float] = []
    for sent in _sentences(answer):
        sent_tags = [int(n) for g in TAG_GROUP.findall(sent) for n in TAG_NUM.findall(g)]
        body = TAG_GROUP.sub(" ", sent)
        stems = content_stems(body)
        if len(stems) < 3:
            continue
        valid = [n for n in sent_tags if n in seen]
        if not valid:
            res.warnings.append(f"uncited_sentence: {body[:60]!r}")
            evidence = all_cited_text
        else:
            evidence = " ".join(passages[n - 1].chunk.text for n in valid)
        ev_stems = content_stems(evidence)
        support = len(stems & ev_stems) / len(stems)
        supports.append(support)
        if support < SUPPORT_THRESHOLD:
            res.issues.append(f"unsupported_claim: only {support:.0%} of {body[:60]!r} is in the cited text")
        missing = numbers(body) - numbers(evidence)
        if missing:
            res.issues.append(f"unsupported_number: {sorted(missing)} not in cited text for {body[:50]!r}")
    res.min_support = min(supports) if supports else None
    if not supports:
        res.issues.append("no_substantive_claim: answer contains no checkable claim")
    return done()
