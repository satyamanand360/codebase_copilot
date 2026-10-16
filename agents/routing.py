"""Turning user input and the router LLM's reply into one of four intents."""
import re
from typing import Optional

INTENTS = ("qa", "review", "test", "docs")
_DIFF_LINE = re.compile(r"^(diff --git |@@ |\+\+\+ |--- )", re.MULTILINE)


def looks_like_diff(text: str) -> bool:
    """Cheap rule that skips the router LLM call for obvious PR diffs."""
    return len(_DIFF_LINE.findall(text)) >= 2


def parse_intent(reply: Optional[str], default: str = "qa") -> str:
    """Map a free-text router reply to an intent.

    Exact matches win; otherwise the first intent word found anywhere in the
    reply is used, so "Review.", "'test'" or "Category: docs" still route
    correctly instead of silently falling back to qa.
    """
    if not reply:
        return default
    cleaned = reply.strip().lower().strip(" .'\"`*")
    if cleaned in INTENTS:
        return cleaned
    words = re.findall(r"[a-z]+", reply.lower())
    for w in words:
        if w in INTENTS:
            return w
    return default
