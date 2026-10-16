"""Reciprocal Rank Fusion, kept dependency-free so it can be unit tested."""
from typing import Dict, List, Sequence


def rrf_fuse(rankings: Sequence[Sequence[str]], k: int = 60, top_k: int = None) -> List[str]:
    """Fuse several ranked lists of ids.

    score(d) = sum over lists of 1 / (k + rank(d)), with rank starting at 1.
    Only positions are used, so lists whose raw scores live on different
    scales (cosine vs BM25) can be combined without normalisation.
    Ties keep the order in which ids were first seen.
    """
    scores: Dict[str, float] = {}
    first_seen: Dict[str, int] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            first_seen.setdefault(doc_id, len(first_seen))
    fused = sorted(scores, key=lambda d: (-scores[d], first_seen[d]))
    return fused[:top_k] if top_k else fused
