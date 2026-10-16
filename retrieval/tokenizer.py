"""Code-aware tokenizer for BM25.

Plain ``text.lower().split()`` turns ``self.add(a, 0)`` into tokens such as
``self.add(a,`` that never match a query word like ``add``. This tokenizer
extracts identifiers and numbers, keeps each full identifier, and also emits
its snake_case / camelCase parts so ``main_calculation`` matches "calculation"
and ``HybridSearcher`` matches "searcher".

The same function must be used at index time and at query time.
"""
import re
from typing import List

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
_CAMEL = re.compile(r"([a-z0-9])([A-Z])")


def code_tokens(text: str) -> List[str]:
    tokens: List[str] = []
    for tok in _IDENT.findall(text):
        lowered = tok.lower()
        tokens.append(lowered)
        parts = _CAMEL.sub(r"\1_\2", tok).lower().split("_")
        tokens.extend(p for p in parts if p and p != lowered)
    return tokens
