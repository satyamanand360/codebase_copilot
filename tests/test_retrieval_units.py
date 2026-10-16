import json

from retrieval.fusion import rrf_fuse
from retrieval.multi_hop import MultiHopExpander
from retrieval.tokenizer import code_tokens


# ---------------------------------------------------------------- tokenizer
def test_tokenizer_splits_code_punctuation():
    toks = code_tokens("return self.add(a, 0) * b")
    assert "add" in toks and "self" in toks and "0" in toks
    assert not any("(" in t or "." in t for t in toks)


def test_tokenizer_splits_snake_and_camel_case():
    toks = code_tokens("main_calculation HybridSearcher")
    assert {"main_calculation", "main", "calculation"} <= set(toks)
    assert {"hybridsearcher", "hybrid", "searcher"} <= set(toks)


def test_tokenizer_matches_query_word_that_whitespace_split_missed():
    # The original lower().split() produced 'multiply(self,' which never matched 'multiply'
    assert "multiply" in code_tokens("def multiply(self, a, b):")


# ------------------------------------------------------------------- fusion
def test_rrf_rewards_documents_found_by_both_lists():
    fused = rrf_fuse([["a", "b", "c"], ["c", "d", "a"]], k=60)
    assert fused[0] == "a"  # rank 1 + rank 3 beats c (rank 3 + rank 1) on tie-break by first seen
    assert set(fused) == {"a", "b", "c", "d"}


def test_rrf_scores_follow_formula():
    fused = rrf_fuse([["x", "y"], ["y"]], k=60)
    # y: 1/62 + 1/61  >  x: 1/61
    assert fused == ["y", "x"]


def test_rrf_top_k_and_empty_lists():
    assert rrf_fuse([[], []]) == []
    assert rrf_fuse([["a", "b", "c"]], top_k=2) == ["a", "b"]


# ---------------------------------------------------------------- multi-hop
def _chunk(name, path="m.py"):
    return {"chunk_id": f"{path}::{name}", "symbol_name": name, "file_path": path}


def _expander(tmp_path, graph, **kw):
    p = tmp_path / "g.json"
    p.write_text(json.dumps(graph))
    return MultiHopExpander(graph_path=str(p), **kw)


def test_multi_hop_adds_callees_and_callers(tmp_path):
    graph = {"multiply": {"calls": ["add"], "called_by": ["main"]},
             "add": {"calls": [], "called_by": ["multiply"]},
             "main": {"calls": ["multiply", "print"], "called_by": []}}
    all_chunks = [_chunk("add"), _chunk("multiply"), _chunk("main")]
    out = _expander(tmp_path, graph).expand([all_chunks[1]], all_chunks)
    assert [c["symbol_name"] for c in out] == ["multiply", "add", "main"]


def test_multi_hop_caps_fan_out(tmp_path):
    callers = [f"f{i}" for i in range(10)]
    graph = {"util": {"calls": [], "called_by": callers}}
    all_chunks = [_chunk("util")] + [_chunk(n) for n in callers]
    out = _expander(tmp_path, graph, max_neighbors=3).expand([all_chunks[0]], all_chunks)
    assert len(out) == 1 + 3


def test_multi_hop_skips_ambiguous_names(tmp_path):
    graph = {"run": {"calls": ["__init__"], "called_by": []}}
    all_chunks = [_chunk("run")] + [_chunk("__init__", path=f"f{i}.py") for i in range(3)]
    out = _expander(tmp_path, graph, max_defs_per_name=2).expand([all_chunks[0]], all_chunks)
    assert len(out) == 1
