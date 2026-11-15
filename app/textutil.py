"""Small text helpers shared by retrieval gating, verification and evaluation."""
from __future__ import annotations

import re
from functools import lru_cache

import snowballstemmer

# Measured on real gpt-oss-120b prompts (3 samples so far, math-heavy text): 1.4-1.65 tokens/word.
# Used for chunk sizing and request-size estimates; the report states the final measured ratio.
TOKENS_PER_WORD = 1.5

STOP = frozenset(
    "a an the of in on for to and or is are was were be been being by with as at from that this these those "
    "it its which what how does do did why when where who whom their they them we our you your can could "
    "would should may might will shall has have had having not no nor than then there here also into about "
    "between over under such each both other more most some any only own same so very just using used use "
    "paper papers model models described describe state states "
    # words that phrase a question but carry no topic
    "differ differs difference differences different compare compared comparison treat treats explain "
    "discuss report reports reported propose proposes author authors achieve achieves decide decides "
    "purpose main mean means called work works way kind".split()
)
_TOK = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
NUM_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
    "seven": "7", "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12",
    "sixteen": "16", "thirty": "30", "sixty": "60", "hundred": "100",
}


_STEMMER = snowballstemmer.stemmer("english")


@lru_cache(maxsize=50_000)
def stem(w: str) -> str:
    """Snowball (Porter2) stem; maps e.g. practical/practice -> practic, heads -> head."""
    return _STEMMER.stemWord(w)


def content_stems(text: str) -> set[str]:
    """Stemmed non-stopword tokens; number words are mapped to digits."""
    out: set[str] = set()
    for t in _TOK.findall(text.lower().replace("-", " ")):
        t = NUM_WORDS.get(t, t)
        if t in STOP or len(t) < 2 and not t.isdigit():
            continue
        out.add(t if t[0].isdigit() else stem(t))
    return out


def numbers(text: str) -> set[str]:
    """Numeric tokens in ``text`` (digits, decimals; number words mapped to digits)."""
    # thousands groups: "100,000" / "100 000" -> "100000" (models often use a space separator)
    text = re.sub(r"(?<![\d.])(\d{1,3})(?:[ ,](\d{3}))+(?![\d])",
                  lambda m: re.sub(r"[ ,]", "", m.group(0)), text)
    out = set(re.findall(r"(?<![\w.])\d+(?:\.\d+)?", text.replace(",", "")))
    for w in re.findall(r"[a-z]+", text.lower()):
        if w in NUM_WORDS:
            out.add(NUM_WORDS[w])
    return out
