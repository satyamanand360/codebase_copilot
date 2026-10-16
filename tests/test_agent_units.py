import os

from agents.common import format_context, strip_code_fences
from agents.diff_query import diff_to_query
from agents.routing import looks_like_diff, parse_intent
from agents.sandbox import SandboxExecutor

DIFF = """diff --git a/retrieval/fusion.py b/retrieval/fusion.py
--- a/retrieval/fusion.py
+++ b/retrieval/fusion.py
@@ -10,7 +10,7 @@ def rrf_fuse(rankings, k=60, top_k=None):
-            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
+            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k + rank)
"""


# ------------------------------------------------------------------ routing
def test_parse_intent_exact_and_messy_replies():
    assert parse_intent("review") == "review"
    assert parse_intent("Review.") == "review"
    assert parse_intent("'test'") == "test"
    assert parse_intent("Category: docs") == "docs"
    assert parse_intent("  QA  ") == "qa"


def test_parse_intent_defaults_to_qa():
    assert parse_intent("I am not sure") == "qa"
    assert parse_intent("") == "qa"
    assert parse_intent(None) == "qa"


def test_looks_like_diff():
    assert looks_like_diff(DIFF)
    assert not looks_like_diff("What does rrf_fuse do?")


# --------------------------------------------------------------- diff query
def test_diff_query_extracts_file_function_and_identifiers():
    q = diff_to_query(DIFF)
    assert "retrieval/fusion.py" in q
    assert "rrf_fuse" in q
    assert "weight" in q and "scores" in q
    assert "diff --git" not in q


def test_diff_query_falls_back_for_plain_text():
    assert diff_to_query("just a sentence") == "just a sentence"


# ------------------------------------------------------------------- common
def test_format_context_uses_qualified_name_and_lines():
    ctx = format_context([{"file_path": "a.py", "start_line": 3, "end_line": 5,
                           "symbol_name": "add", "qualified_name": "Math.add", "content": "def add(): pass"}])
    assert "File: a.py (lines 3-5)" in ctx and "Symbol: Math.add" in ctx


def test_strip_code_fences():
    assert strip_code_fences("```python\nx = 1\n```") == "x = 1\n"
    assert strip_code_fences("x = 1") == "x = 1\n"


# ------------------------------------------------------------------ sandbox
def test_sandbox_reports_pass_and_fail():
    sb = SandboxExecutor(timeout=60)
    assert sb.execute_test("def test_ok():\n    assert 1 + 1 == 2\n")["passed"]
    failed = sb.execute_test("def test_bad():\n    assert 1 == 2\n")
    assert not failed["passed"] and "assert" in failed["output"]


def test_sandbox_does_not_leak_secrets(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "super-secret")
    code = "import os\ndef test_env():\n    assert 'GOOGLE_API_KEY' not in os.environ\n"
    assert SandboxExecutor(timeout=60).execute_test(code)["passed"]
