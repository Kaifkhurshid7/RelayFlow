"""BM25 retrieval with an optional lightweight lexical reranker.

Design notes
- One BM25 index per chunk size; indexes are cheap to build (<1s) so they are
  built in memory rather than pickled.
- Reranking (config C) fetches ``RERANK_POOL`` x top_k BM25 candidates and
  rescores them by query-term coverage, bigram matches and section-title overlap.
  No neural model is required; a semantic reranker can be added behind the same
  ``rerank`` flag if measurement shows it is worth the cost.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from app.config import RETRIEVAL_CONFIGS, RetrievalConfig
from app.retrieval.paper_ingestion import Chunk, build_chunks
from app.textutil import content_stems, stem

RERANK_POOL = 3
TITLE_BOOST = 3  # section title is indexed this many times (field boost)

_STOP = frozenset(
    "a an the of in on for to and or is are was were be been by with as at from that this "
    "it its which what how does do did why when where who whom their they we our can used "
    "using use paper papers model models".split()
)
_TOKEN = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Lowercase, stop-word-free, Snowball-stemmed alphanumeric tokens."""
    text = re.sub(r"'s\b", "", text.lower().replace("\u2019", "'"))  # possessives: Nvidia's -> nvidia
    return [stem(t) for t in _TOKEN.findall(text) if t not in _STOP]


@dataclass
class Passage:
    """A retrieved chunk plus its retrieval score and rank."""

    chunk: Chunk
    score: float
    rank: int

    def citation(self) -> str:
        c = self.chunk
        pages = f"p.{c.page_start}" if c.page_start == c.page_end else f"pp.{c.page_start}-{c.page_end}"
        return f"{c.paper_title} (arXiv:{c.arxiv_id}{c.version}), {c.section}, {pages}"


class Retriever:
    """BM25 retriever over one chunking of the library."""

    def __init__(self, chunks: list[Chunk], config: RetrievalConfig) -> None:
        self.config = config
        self.chunks = chunks
        self._tokens = [tokenize(c.section) * TITLE_BOOST + tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(self._tokens)
        # stem-level document frequencies for the evidence-coverage gate
        self._stem_df: dict[str, int] = {}
        for c in chunks:
            for t in content_stems(f"{c.section} {c.text}"):
                self._stem_df[t] = self._stem_df.get(t, 0) + 1

    @classmethod
    def from_config(cls, name: str) -> "Retriever":
        cfg = RETRIEVAL_CONFIGS[name]
        return cls(build_chunks(cfg.chunk_tokens), cfg)

    def search(
        self,
        query: str,
        top_k: int | None = None,
        paper_ids: set[str] | None = None,
    ) -> list[Passage]:
        """Return the best ``top_k`` passages (defaults to the config's top_k).

        ``paper_ids`` restricts the search to specific papers (e.g. from question
        analysis); ``None`` searches the whole library.
        """
        k = top_k or self.config.top_k
        q = tokenize(query)
        if not q:
            return []
        scores = self._bm25.get_scores(q)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        if paper_ids:
            order = [i for i in order if self.chunks[i].paper_id in paper_ids]
        pool = k * RERANK_POOL if self.config.rerank else k
        order = [i for i in order[:pool] if scores[i] > 0]
        if self.config.rerank:
            order = self._rerank(q, order, scores)[:k]
        return [Passage(self.chunks[i], float(scores[i]), r) for r, i in enumerate(order[:k], 1)]

    def _rerank(self, q: list[str], idxs: list[int], bm25: "list[float]") -> list[int]:
        qset, qbig = set(q), set(zip(q, q[1:]))
        top = max((bm25[i] for i in idxs), default=1.0) or 1.0

        def score(i: int) -> float:
            toks = self._tokens[i]
            tset = set(toks)
            coverage = len(qset & tset) / len(qset)
            bigrams = len(qbig & set(zip(toks, toks[1:]))) / len(qbig) if qbig else 0.0
            title = len(qset & set(tokenize(self.chunks[i].section))) / len(qset)
            return bm25[i] / top + 0.5 * coverage + 0.3 * bigrams + 0.2 * title

        return sorted(idxs, key=score, reverse=True)

    def search_balanced(self, query: str, paper_ids: list[str], top_k: int | None = None) -> list[Passage]:
        """Cross-paper retrieval: loop over papers, retrieve per paper, interleave round-robin.

        A single global ranking can fill every slot from the paper that happens to
        share the most vocabulary with the question; this guarantees each named
        paper is represented whenever ``top_k >= len(paper_ids)``.
        """
        k = top_k or self.config.top_k
        per = max(1, math.ceil(k / max(1, len(paper_ids))))
        lists = [self.search(query, top_k=per, paper_ids={pid}) for pid in paper_ids]
        merged: list[Passage] = []
        for r in range(per):
            for lst in lists:
                if r < len(lst):
                    merged.append(lst[r])
        return [Passage(p.chunk, p.score, i) for i, p in enumerate(merged[:k], 1)]

    def coverage(self, query: str, passages: list[Passage]) -> float:
        """Idf-weighted share of the question's content terms found in ``passages`` (0..1).

        Terms absent from the whole library get the maximum idf (they are the most
        informative terms: if the question asks about "revenue" and no paper ever
        mentions revenue, that must count heavily against answerability).
        """
        q = content_stems(query)
        if not q:
            return 0.0
        have: set[str] = set()
        for p in passages:
            have |= content_stems(f"{p.chunk.section} {p.chunk.text}")
        n = len(self.chunks)

        def w(t: str) -> float:
            return math.log(1 + n / self._stem_df.get(t, 0.5))

        total = sum(w(t) for t in q)
        return sum(w(t) for t in q if t in have) / total if total else 0.0
