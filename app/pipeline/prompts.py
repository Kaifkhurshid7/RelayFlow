"""Prompt construction. The source block format is parsed by the stand-in
provider and by tests, so change it in one place only."""
from __future__ import annotations

from app.retrieval.retrieval import Passage

REFUSAL = "The provided research library does not contain sufficient information to answer this question."

SYSTEM_PROMPT = f"""You answer questions about a fixed library of three research papers using ONLY the numbered sources provided in the user message.
Rules:
1. Use only facts stated in the sources. Never use outside or general knowledge, even if you know the answer.
2. Cite every claim with source tags such as [S1] or [S2][S3], placed immediately after the claim. Use plain ASCII square brackets only (never 【 】 or other bracket characters).
3. If the sources do not contain enough information to answer, reply with exactly this sentence and nothing else: {REFUSAL}
4. If the question's premise contradicts the sources, say so and cite the sources.
5. Do not cite a source unless it directly supports the claim. Quote numbers exactly as written in the sources.
6. Be concise: at most 120 words."""


def source_header(i: int, p: Passage) -> str:
    c = p.chunk
    pages = f"p.{c.page_start}" if c.page_start == c.page_end else f"pp.{c.page_start}-{c.page_end}"
    return f"[S{i}] ({c.paper_title} | arXiv:{c.arxiv_id} {c.version} | {c.section} | {pages})"


def build_messages(
    question: str,
    passages: list[Passage],
    repair_feedback: str | None = None,
    previous_answer: str | None = None,
) -> list[dict[str, str]]:
    """System + user messages; optionally a repair turn after failed verification."""
    blocks = [f"{source_header(i, p)}\n{p.chunk.text}" for i, p in enumerate(passages, 1)]
    user = "Sources:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}"
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
    if repair_feedback and previous_answer is not None:
        msgs.append({"role": "assistant", "content": previous_answer})
        msgs.append(
            {
                "role": "user",
                "content": (
                    "Your previous answer failed automatic verification: "
                    f"{repair_feedback} Rewrite it using only the sources, cite each claim with [S#], "
                    f"or reply with exactly: {REFUSAL}"
                ),
            }
        )
    return msgs
