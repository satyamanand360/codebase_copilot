"""HybridSearcher with a fake vector store and fake embeddings (no model downloads)."""
import json

import pytest

pytest.importorskip("rank_bm25")
pytest.importorskip("qdrant_client")
pytest.importorskip("langchain_community")

from retrieval.hybrid_search import HybridSearcher  # noqa: E402

CHUNKS = [
    {"chunk_id": "m.py::MathOperations.add", "symbol_name": "add", "file_path": "m.py",
     "content": "def add(self, a, b):\n    return a + b"},
    {"chunk_id": "m.py::MathOperations.multiply", "symbol_name": "multiply", "file_path": "m.py",
     "content": "def multiply(self, a, b):\n    return self.add(a, 0) * b"},
    {"chunk_id": "m.py::main_calculation", "symbol_name": "main_calculation", "file_path": "m.py",
     "content": "def main_calculation():\n    result = MathOperations().multiply(5, 10)\n    return result"},
]


class FakeEmbeddings:
    def embed_query(self, text):
        return [0.0] * 384


class Hit:
    def __init__(self, cid):
        self.payload = {"chunk_id": cid}


class FakeQdrant:
    def __init__(self, order, fail=False):
        self.order, self.fail = order, fail

    def query_points(self, collection_name, query, limit):
        if self.fail:
            raise RuntimeError("down")
        return type("R", (), {"points": [Hit(c) for c in self.order[:limit]]})()


@pytest.fixture
def index_dir(tmp_path):
    (tmp_path / "chunks.json").write_text(json.dumps(CHUNKS))
    return str(tmp_path)


def make(index_dir, order=(), fail=False):
    return HybridSearcher(qdrant=FakeQdrant(list(order), fail), embeddings=FakeEmbeddings(), index_dir=index_dir)


def test_bm25_finds_identifier(index_dir):
    ids = [c["chunk_id"] for c in make(index_dir).search("multiply", mode="bm25")]
    assert ids[0] == "m.py::MathOperations.multiply"
    assert "m.py::MathOperations.add" not in ids  # shares no token with the query


def test_hybrid_fuses_dense_and_sparse(index_dir):
    # dense: add, multiply   sparse: multiply, main_calculation  -> multiply is in both
    s = make(index_dir, order=["m.py::MathOperations.add", "m.py::MathOperations.multiply"])
    ids = [c["chunk_id"] for c in s.search("multiply")]
    assert ids[0] == "m.py::MathOperations.multiply"  # found by both lists


def test_dense_failure_falls_back_to_bm25(index_dir):
    s = make(index_dir, fail=True)
    ids = [c["chunk_id"] for c in s.search("multiply")]
    assert ids and ids[0] == "m.py::MathOperations.multiply"
    assert s.dense_available is False
