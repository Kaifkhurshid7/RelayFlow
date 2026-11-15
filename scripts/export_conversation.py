"""Export the genuine development conversation (user prompts) from the Claude Code transcript.

Reads the session transcript (JSONL), extracts every human message verbatim with its timestamp, and
redacts anything that looks like a credential before writing ``docs/conversation_history.md``.
Nothing is paraphrased or invented: assistant messages are summarised elsewhere (docs/HOW_I_WORKED.md)
from the concrete events in this repository, and clearly marked as such.

Usage:
  python scripts/export_conversation.py [path/to/session.jsonl]
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT = Path.home() / ".claude" / "projects" / "e--RelayFlow"
SECRETS = [
    (re.compile(r"gsk_[A-Za-z0-9]{20,}"), "gsk_[REDACTED]"),
    (re.compile(r"\bAQ\.[A-Za-z0-9_\-]{20,}"), "AQ.[REDACTED]"),
    (re.compile(r"AIza[0-9A-Za-z_\-]{20,}"), "AIza[REDACTED]"),
    (re.compile(r"(?i)(api[_-]?key\s*[=:]\s*)\S{16,}"), r"\1[REDACTED]"),
]
NOISE = re.compile(r"<(system-reminder|ide_opened_file|ide_selection)>.*?</\1>", re.S)


def redact(s: str) -> str:
    for rx, rep in SECRETS:
        s = rx.sub(rep, s)
    return s


def human_messages(path: Path) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("type") != "user":
            continue
        c = d.get("message", {}).get("content")
        parts = [c] if isinstance(c, str) else [p.get("text", "") for p in (c or []) if p.get("type") == "text"]
        for txt in parts:
            txt = NOISE.sub("", txt).strip()
            if txt and not txt.startswith("<task-notification>") and "[SYSTEM NOTIFICATION" not in txt:
                out.append((d.get("timestamp", ""), txt))
    return out


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else max(DEFAULT.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    msgs = human_messages(src)
    lines = [
        "# Conversation history (user prompts, verbatim)",
        "",
        f"Source: Claude Code session transcript `{src.name}`. {len(msgs)} human messages, in order.",
        "Credentials that appeared in the chat were redacted by `scripts/export_conversation.py`.",
        "Messages the user pasted from elsewhere are shown as received. Assistant replies are NOT reproduced",
        "here; see `docs/HOW_I_WORKED.md` for the decisions, mistakes and corrections, each tied to a file or",
        "test in this repository.",
        "",
    ]
    for i, (ts, txt) in enumerate(msgs, 1):
        lines += [f"## Prompt {i}  ({ts})", "", "```text", redact(txt), "```", ""]
    out = ROOT / "docs" / "conversation_history.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    leaks = [m for m in re.findall(r"gsk_[A-Za-z0-9]{10,}|AQ\.[A-Za-z0-9_\-]{10,}", out.read_text(encoding="utf-8"))
             if "REDACTED" not in m]
    print(f"wrote {out} ({len(msgs)} prompts); leaked-credential patterns remaining: {len(leaks)}")


if __name__ == "__main__":
    main()
