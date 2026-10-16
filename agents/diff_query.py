"""Build a retrieval query from a unified diff.

The first 500 raw characters of a diff are mostly headers and +/- markers,
which make a poor search query. Instead, collect the changed files, the
function names from hunk headers, and the identifiers on changed lines.
"""
import re
from typing import List

_FILE = re.compile(r"^\+\+\+ b/(.+)$", re.MULTILINE)
_HUNK_CTX = re.compile(r"^@@[^@]*@@\s*(?:async\s+)?(?:def|class)\s+(\w+)", re.MULTILINE)
_DEF = re.compile(r"^[+-]\s*(?:async\s+)?(?:def|class)\s+(\w+)", re.MULTILINE)
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
_STOP = {"self", "def", "class", "return", "import", "from", "None", "True", "False", "print", "and", "not", "for", "the"}


def diff_to_query(diff: str, max_terms: int = 40) -> str:
    terms: List[str] = []

    def add(t: str):
        if t not in terms and t not in _STOP:
            terms.append(t)

    for path in _FILE.findall(diff):
        add(path)
    for name in _HUNK_CTX.findall(diff) + _DEF.findall(diff):
        add(name)
    for line in diff.splitlines():
        if line.startswith(("+", "-")) and not line.startswith(("+++", "---")):
            for ident in _IDENT.findall(line[1:]):
                add(ident)
    if not terms:  # not a diff after all
        return diff[:500]
    return " ".join(terms[:max_terms])
