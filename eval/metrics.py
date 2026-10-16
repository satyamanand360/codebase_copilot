"""Evaluation helpers with no heavy dependencies (unit tested)."""
from typing import Iterable, List, Optional, Sequence


def recall_at_k(retrieved: Sequence[str], gold: Iterable[str], k: int) -> float:
    """Fraction of gold chunk ids that appear in the first k retrieved ids."""
    gold = set(gold)
    if not gold:
        return 0.0
    return len(gold & set(retrieved[:k])) / len(gold)


def hit_at_k(retrieved: Sequence[str], gold: Iterable[str], k: int) -> float:
    """1.0 if any gold id is in the first k, else 0.0."""
    return 1.0 if set(gold) & set(retrieved[:k]) else 0.0


def reciprocal_rank(retrieved: Sequence[str], gold: Iterable[str]) -> float:
    """1 / rank of the first gold id (0 if none retrieved). Averaged over questions it is MRR."""
    gold = set(gold)
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in gold:
            return 1.0 / rank
    return 0.0


def mean(values: List[Optional[float]]) -> Optional[float]:
    """Mean of the non-None values; None if there are none."""
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def parse_score(text: str) -> Optional[float]:
    """Parse an LLM judge reply like '0.85' into a float in [0, 1].

    Returns None when the reply can't be parsed, so failures are counted
    instead of being silently recorded as 0.5.
    """
    if not text:
        return None
    for token in text.replace(",", " ").split():
        try:
            value = float(token.strip(" .:*`'\""))
        except ValueError:
            continue
        if 0.0 <= value <= 1.0:
            return value
    return None
