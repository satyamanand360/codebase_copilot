import json
import os

from eval.metrics import hit_at_k, mean, parse_score, recall_at_k, reciprocal_rank

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_recall_and_hit():
    retrieved = ["a", "b", "c", "d"]
    assert recall_at_k(retrieved, ["a", "d"], 2) == 0.5
    assert recall_at_k(retrieved, ["a", "d"], 4) == 1.0
    assert hit_at_k(retrieved, ["c"], 2) == 0.0
    assert hit_at_k(retrieved, ["c"], 3) == 1.0


def test_reciprocal_rank():
    assert reciprocal_rank(["x", "y", "z"], ["z"]) == 1 / 3
    assert reciprocal_rank(["x"], ["q"]) == 0.0


def test_parse_score_never_invents_a_default():
    assert parse_score("0.85") == 0.85
    assert parse_score("Score: 0.7.") == 0.7
    assert parse_score("I think it is mostly faithful") is None
    assert parse_score("") is None
    assert parse_score("7") is None  # out of range


def test_mean_ignores_none():
    assert mean([1.0, None, 0.0]) == 0.5
    assert mean([None]) is None


def test_benchmark_shape():
    with open(os.path.join(ROOT, "eval", "benchmark.json"), encoding="utf-8") as f:
        bench = json.load(f)
    assert len(bench) >= 30
    for item in bench:
        assert item["question"] and item["ground_truth"] and item["gold_chunks"]
