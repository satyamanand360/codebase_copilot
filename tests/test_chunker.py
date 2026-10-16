import json
import os

import pytest

pytest.importorskip("tree_sitter")
pytest.importorskip("tree_sitter_python")

from ingestion.chunker import CodeChunker  # noqa: E402
from ingestion.collect import collect_chunks, iter_python_files  # noqa: E402
from ingestion.graph_builder import DependencyGraph  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DUMMY = os.path.join(ROOT, "dummy_repo", "test_dummy.py")


@pytest.fixture(scope="module")
def dummy_chunks():
    return {c["qualified_name"]: c for c in CodeChunker().chunk_file(DUMMY, rel_path="dummy_repo/test_dummy.py")}


def test_chunk_ids_are_qualified(dummy_chunks):
    assert set(dummy_chunks) == {"<module>", "MathOperations", "MathOperations.add",
                                 "MathOperations.multiply", "main_calculation"}
    assert dummy_chunks["MathOperations.add"]["chunk_id"] == "dummy_repo/test_dummy.py::MathOperations.add"


def test_line_numbers_are_one_based(dummy_chunks):
    m = dummy_chunks["MathOperations.multiply"]
    assert (m["start_line"], m["end_line"]) == (5, 6)
    assert m["content"].startswith("def multiply")


def test_calls_exclude_object_names(dummy_chunks):
    assert dummy_chunks["MathOperations.multiply"]["calls"] == ["add"]
    assert set(dummy_chunks["main_calculation"]["calls"]) == {"MathOperations", "multiply", "print"}


def test_class_chunk_does_not_repeat_method_bodies(dummy_chunks):
    cls = dummy_chunks["MathOperations"]
    assert "return a + b" not in cls["content"]
    assert "# methods: add, multiply" in cls["content"]


def test_module_chunk_holds_top_level_code(dummy_chunks):
    mod = dummy_chunks["<module>"]
    assert "__main__" in mod["content"] and "main_calculation" in mod["calls"]


def test_decorators_are_included(tmp_path):
    src = tmp_path / "d.py"
    src.write_text("import functools\n\n@functools.lru_cache\ndef cached(x):\n    return helper(x)\n")
    chunks = {c["qualified_name"]: c for c in CodeChunker().chunk_file(str(src), rel_path="d.py")}
    assert chunks["cached"]["content"].startswith("@functools.lru_cache")
    assert chunks["cached"]["start_line"] == 3
    assert chunks["cached"]["calls"] == ["helper"]


def test_walk_finds_files_when_run_on_current_dir(monkeypatch):
    # Regression: the old filter skipped every folder when the root was "."
    monkeypatch.chdir(ROOT)
    files = {os.path.normpath(p) for p in iter_python_files(".")}
    assert os.path.normpath("./agents/qa_agent.py") in files
    assert not any(part.startswith(".") and part != "." for p in files for part in p.split(os.sep))


def test_graph_has_no_self_nodes(dummy_chunks):
    g = DependencyGraph()
    for c in dummy_chunks.values():
        g.add_chunk(c)
    assert "self" not in g.graph and "math_ops" not in g.graph


def test_every_benchmark_gold_chunk_exists(monkeypatch):
    monkeypatch.setenv("INDEX_SKIP_DIRS", "tests,scripts")
    ids = {c["chunk_id"] for c in collect_chunks(ROOT)}
    with open(os.path.join(ROOT, "eval", "benchmark.json"), encoding="utf-8") as f:
        bench = json.load(f)
    missing = sorted({g for q in bench for g in q["gold_chunks"] if g not in ids})
    assert not missing, missing
