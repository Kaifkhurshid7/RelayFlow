"""PDF extraction and chunking with page + section metadata.

Heading detection uses font weight (bold flag) rather than text regexes alone,
because two of the three papers print the section number and title as separate
lines. Everything after a ``References`` heading is flagged ``is_reference`` and
is excluded from the retrieval index (bibliographies are not evidence); it
resumes as normal text at the first appendix heading.

Token counts are estimated as ``words * TOKENS_PER_WORD`` (app.textutil) to avoid a tokenizer
dependency. This is an approximation and is documented as such in the report.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from dataclasses import asdict, dataclass
from pathlib import Path

import pymupdf

from app.config import LIBRARY, Paper, get_settings
from app.textutil import TOKENS_PER_WORD

OVERLAP_FRACTION = 0.10

_NUM_ONLY = re.compile(r"^(\d+(\.\d+)*\.?|[A-Z](\.\d+)*)$")
_NUM_TITLE = re.compile(r"^(\d+(\.\d+)*\.?|[A-Z](\.\d+)+)\s+\S.*$")
_NAMED = re.compile(r"^(Abstract|References|Acknowledg\w*|Appendix\b.*)$", re.I)
_BOLD = 16  # PyMuPDF span flag bit
_THREE_LEVEL = re.compile(r"^\d+\.\d+\.\d+\s+[A-Z]\S*(\s+\S+)*$")


@dataclass
class Line:
    page: int
    text: str
    section: str
    is_reference: bool


@dataclass
class Chunk:
    chunk_id: str
    paper_id: str
    paper_title: str
    arxiv_id: str
    version: str
    page_start: int
    page_end: int
    section: str
    text: str
    n_tokens_est: int


def est_tokens(text: str) -> int:
    """Approximate token count (words * 1.3)."""
    return round(len(text.split()) * TOKENS_PER_WORD)


def _line_text(line: dict) -> str:
    return "".join(s["text"] for s in line["spans"]).strip()


def _is_bold(line: dict) -> bool:
    spans = [s for s in line["spans"] if s["text"].strip()]
    return bool(spans) and all(s["flags"] & _BOLD for s in spans)


def _is_horizontal(line: dict) -> bool:
    dx, dy = line["dir"]
    return abs(dx) > 0.99 and abs(dy) < 0.05


@dataclass
class _HeadingState:
    """Tracks section numbering so bold table cells are not mistaken for headings."""

    last_major: int = 0
    last_appendix: str = ""  # last accepted appendix letter
    in_refs: bool = False
    seen_refs: bool = False

    def accept(self, heading: str) -> bool:
        """Return True if ``heading`` is a plausible next heading; updates state."""
        num, _, title = heading.partition(" ")
        if _NAMED.match(heading):
            m = re.match(r"Appendix\s+([A-Z])\b", heading)
            if m:
                self.last_appendix = m.group(1)
            return True
        if not title or not title[0].isalpha():
            return False  # e.g. "3.3 x 10^18", "51.0 59.9/72.6", "55.5 #FLOPs"
        if num[0].isdigit():
            if self.seen_refs:
                return False  # numbered text after References is not a heading
            major = int(num.split(".")[0])
            if major not in (self.last_major, self.last_major + 1):
                return False
            self.last_major = major
            return True
        letter = num[0]
        if not self.seen_refs:
            return False
        nxt = chr(ord(self.last_appendix or "@") + 1)
        if letter not in (self.last_appendix, nxt):
            return False
        self.last_appendix = letter
        return True


def extract_lines(pdf_path: Path) -> list[Line]:
    """Return body lines in reading order, tagged with section and reference flag."""
    out: list[Line] = []
    section = "Front matter"
    state = _HeadingState()
    in_refs = False
    with pymupdf.open(pdf_path) as doc:
        for pno, page in enumerate(doc, start=1):
            pending_num: str | None = None
            for block in page.get_text("dict")["blocks"]:
                for line in block.get("lines", []):
                    if not _is_horizontal(line):
                        continue  # drops the rotated arXiv margin stamp
                    text = _line_text(line)
                    if not text:
                        continue
                    heading: str | None = None
                    if _is_bold(line) and len(text) < 100:
                        if _NUM_TITLE.match(text) or _NAMED.match(text):
                            heading, pending_num = text, None
                        elif pending_num and not _NUM_ONLY.match(text):
                            heading, pending_num = f"{pending_num} {text}", None
                        elif _NUM_ONLY.match(text):
                            pending_num = text.rstrip(".")
                            continue
                    elif _THREE_LEVEL.match(text) and len(text) < 100 and len(
                        {sp["font"] for sp in line["spans"] if sp["text"].strip()}
                    ) == 1:
                        # Moshi prints x.y.z headings in a small-caps font, not bold.
                        heading, pending_num = text, None
                    else:
                        pending_num = None
                    if heading and state.accept(heading):
                        section = heading
                        low = heading.lower()
                        if low.startswith("references"):
                            in_refs = state.seen_refs = True
                        elif in_refs and re.match(r"^(appendix|[a-z][\s.])", low):
                            in_refs = False
                        continue
                    if text.isdigit():
                        continue  # bare page number
                    out.append(Line(pno, text, section, in_refs))
    return out


def _group_sections(lines: list[Line]) -> list[list[Line]]:
    groups: list[list[Line]] = []
    for ln in lines:
        if groups and groups[-1][0].section == ln.section and groups[-1][0].is_reference == ln.is_reference:
            groups[-1].append(ln)
        else:
            groups.append([ln])
    return groups


def chunk_paper(paper: Paper, lines: list[Line], chunk_tokens: int) -> list[Chunk]:
    """Split each section into ~``chunk_tokens`` windows; never cross sections.

    Reference-section lines are dropped. Windows overlap by ``OVERLAP_FRACTION``.
    """
    max_words = max(20, int(chunk_tokens / TOKENS_PER_WORD))
    step = max(1, int(max_words * (1 - OVERLAP_FRACTION)))
    chunks: list[Chunk] = []
    n = 0
    for group in _group_sections([l for l in lines if not l.is_reference]):
        words: list[tuple[str, int]] = [(w, l.page) for l in group for w in l.text.split()]
        start = 0
        while start < len(words):
            window = words[start : start + max_words]
            text = " ".join(w for w, _ in window)
            chunks.append(
                Chunk(
                    chunk_id=f"{paper.paper_id}-{chunk_tokens}-{n:04d}",
                    paper_id=paper.paper_id,
                    paper_title=paper.title,
                    arxiv_id=paper.arxiv_id,
                    version=paper.version,
                    page_start=window[0][1],
                    page_end=window[-1][1],
                    section=group[0].section,
                    text=text,
                    n_tokens_est=est_tokens(text),
                )
            )
            n += 1
            if start + max_words >= len(words):
                break
            start += step
    return chunks


@lru_cache(maxsize=8)
def _cached_lines(path: str) -> list[Line]:
    """PDF extraction is the slow part; do it once per file per process."""
    return extract_lines(Path(path))


def build_chunks(chunk_tokens: int) -> list[Chunk]:
    """Chunk the whole library at one size."""
    settings = get_settings()
    all_chunks: list[Chunk] = []
    for paper in LIBRARY:
        lines = _cached_lines(str(settings.papers_dir / paper.filename))
        all_chunks.extend(chunk_paper(paper, lines, chunk_tokens))
    return all_chunks


def save_extraction() -> None:
    """Persist extracted lines and chunks (300/600) as JSON for inspection."""
    settings = get_settings()
    settings.extracted_dir.mkdir(parents=True, exist_ok=True)
    for paper in LIBRARY:
        lines = extract_lines(settings.papers_dir / paper.filename)
        (settings.extracted_dir / f"{paper.paper_id}.json").write_text(
            json.dumps([asdict(l) for l in lines], indent=1), encoding="utf-8"
        )
    for size in (300, 600):
        chunks = build_chunks(size)
        (settings.extracted_dir / f"chunks_{size}.json").write_text(
            json.dumps([asdict(c) for c in chunks], indent=1), encoding="utf-8"
        )
        print(f"chunk_tokens={size}: {len(chunks)} chunks")


if __name__ == "__main__":
    save_extraction()
