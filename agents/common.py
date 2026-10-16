"""Retrieval and formatting shared by every agent."""
from typing import Dict, List


def retrieve_context(searcher, reranker, expander, query: str, top_k: int = 30, keep: int = 5) -> List[Dict]:
    """Hybrid search -> cross-encoder rerank -> one-hop graph expansion."""
    candidates = searcher.search(query, top_k=top_k)
    best = reranker.rerank(query, candidates, top_k=keep)
    return expander.expand(best, searcher.chunks)


def format_context(chunks: List[Dict]) -> str:
    blocks = []
    for c in chunks:
        blocks.append(
            f"File: {c['file_path']} (lines {c['start_line']}-{c['end_line']})\n"
            f"Symbol: {c.get('qualified_name', c['symbol_name'])}\n"
            f"Code:\n```python\n{c['content']}\n```"
        )
    return "\n\n".join(blocks)


def strip_code_fences(text: str) -> str:
    """LLMs often wrap code in ```python fences even when told not to."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else ""
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip() + "\n"
