"""Download and verify the exact arXiv paper versions.

Verification: arXiv serves ``/pdf/<id>v<N>`` for any existing version, so a 200
alone is not proof. We additionally (1) check the ``%PDF`` magic bytes,
(2) check the abs page ``/abs/<id>v<N>`` lists that version, and (3) check the
title appears on page 1 of the PDF. If anything fails we stop; we never fall
back to another version.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import pymupdf
import httpx

from app.config import LIBRARY, Paper, get_settings


class VerificationError(RuntimeError):
    """Raised when a downloaded file is not the requested paper version."""


def _norm(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())


def verify_pdf(paper: Paper, path: Path) -> dict[str, object]:
    """Check magic bytes and that page 1 contains the expected title."""
    data = path.read_bytes()
    if not data.startswith(b"%PDF"):
        raise VerificationError(f"{paper.filename}: not a PDF")
    with pymupdf.open(path) as doc:
        first = doc[0].get_text()
        pages = doc.page_count
    if _norm(paper.title) not in _norm(first):
        raise VerificationError(f"{paper.filename}: title not found on page 1")
    # arXiv stamps "arXiv:<id>v<N> [cat] <date>" on page 1 of every PDF.
    stamp = re.search(rf"arXiv:\s*{re.escape(paper.arxiv_id)}(v\d+)", first)
    if stamp is None or stamp.group(1) != paper.version:
        found = stamp.group(1) if stamp else "none"
        raise VerificationError(
            f"{paper.filename}: expected {paper.version}, PDF stamp says {found}"
        )
    return {"stamp": stamp.group(0), "pages": pages, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def verify_version_on_arxiv(client: httpx.Client, paper: Paper) -> None:
    """Confirm the abs page for this exact version exists and names it."""
    r = client.get(f"https://arxiv.org/abs/{paper.arxiv_id}{paper.version}")
    r.raise_for_status()
    if f"[{paper.version}]" not in r.text and f"/{paper.arxiv_id}{paper.version}" not in r.text:
        raise VerificationError(f"{paper.label}: version not confirmed on abs page")


def download_all(force: bool = False) -> dict[str, dict[str, object]]:
    """Download every paper, verify it, and write ``manifest.json``."""
    settings = get_settings()
    settings.papers_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict[str, object]] = {}
    with httpx.Client(follow_redirects=True, timeout=60.0) as client:
        for paper in LIBRARY:
            dest = settings.papers_dir / paper.filename
            verify_version_on_arxiv(client, paper)
            if force or not dest.exists():
                resp = client.get(paper.url)
                resp.raise_for_status()
                dest.write_bytes(resp.content)
            info = verify_pdf(paper, dest)
            manifest[paper.paper_id] = {
                "title": paper.title,
                "arxiv_id": paper.arxiv_id,
                "version": paper.version,
                "url": paper.url,
                "file": paper.filename,
                **info,
            }
            print(f"OK  {paper.label}  {info['pages']} pages  {dest.name}")
    (settings.papers_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    try:
        download_all(force="--force" in sys.argv)
    except (VerificationError, httpx.HTTPError) as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
