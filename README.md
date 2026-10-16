# Codebase Copilot

A multi-agent assistant for Python codebases. Ask questions about the code, get a review of a PR diff, or generate unit tests and documentation. Answers are grounded in retrieved code and cite file paths and line numbers.

## How it works

**Indexing** (`python -m ingestion.indexer .`)

1. **Chunking with Tree-sitter.** Every function and method becomes a chunk (decorators included). Each class gets a header chunk (signature, docstring, attributes, list of methods) so method bodies aren't duplicated, and each file gets a module chunk with its imports and top-level code. Chunk ids are qualified: `path::Class.method`. Line numbers are 1-based.
2. **Dense index.** `all-MiniLM-L6-v2` embeddings (384-d, batched) in a Qdrant collection, recreated on each run so no stale points remain.
3. **Sparse index.** The chunk list is saved to `index/chunks.json`; BM25 is rebuilt from it at load time with a code-aware tokenizer (splits punctuation, snake_case and camelCase).
4. **Call graph.** `index/dependency_graph.json` maps each symbol to the names it calls and the symbols that call it.

**Retrieval** (shared by all agents)

1. Dense top-30 and BM25 top-30 (BM25Plus; chunks sharing no token with the query are dropped).
2. Reciprocal Rank Fusion, `score = Σ 1/(60 + rank)`.
3. Cross-encoder rerank (`cross-encoder/ms-marco-MiniLM-L-6-v2`), keep 5.
4. One-hop call-graph expansion (callees and callers), capped per chunk and skipping ambiguous names.

**Agents** (LangGraph)

A supervisor node routes each request to `qa`, `review`, `test` or `docs`. Obvious diffs go straight to review; otherwise a fast LLM classifies the request and its reply is parsed tolerantly.

- **QA**: generate an answer, verify it against the context with a second LLM call, and on failure regenerate with the verifier's reason (up to 3 attempts).
- **Review**: builds the search query from the diff's files, changed functions and identifiers; the prompt treats the diff as untrusted data.
- **Test**: writes pytest tests. With `ENABLE_TEST_SANDBOX=1` it runs them in a subprocess (temp dir, timeout, no secrets in the environment) and feeds failures back. This executes model-written code on the host, so it is off by default; use a container for untrusted input.
- **Docs**: writes Markdown documentation.

**LLMs**: Gemini is primary, Claude is the fallback (`with_fallbacks`), each with retries. Model names are configurable in `.env`.

**Serving**: FastAPI (`POST /api/chat`, `GET /health`) plus a small web UI in `public/` that sanitises rendered Markdown with DOMPurify. A GitHub Action indexes the PR's code and posts a review comment.

## Quick start

```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                  # add GOOGLE_API_KEY and/or ANTHROPIC_API_KEY

docker compose up -d                                  # optional: Qdrant server (otherwise embedded mode)
python -m ingestion.indexer .                         # index this repo, or pass another path
uvicorn api.main:app --reload                         # open http://localhost:8000
```

## Evaluation

`eval/benchmark.json` has 34 questions about this repository's own code. Each lists the chunk ids that answer it, so retrieval can be scored without an LLM judge.

```bash
INDEX_SKIP_DIRS=tests,scripts python -m ingestion.indexer .
python eval/run_ablation.py        # retrieval metrics per configuration, no API key needed
python eval/run_eval.py --limit 10 # LLM-as-judge: faithfulness, context precision, context recall
```

The ablation runs in CI on every PR that touches retrieval and writes `eval/ablation_results.md`. It compares the original BM25 setup, the code-aware BM25, dense only, hybrid, hybrid + rerank, and hybrid + rerank + multi-hop on Hit@1, Recall@5, MRR and the share of gold chunks that reach the LLM's context.

### Retrieval results (34 questions, 115 chunks, CPU on GitHub Actions)

| Configuration | Hit@1 | Recall@5 | MRR | Context recall | Context size | ms/query |
|---|---|---|---|---|---|---|
| BM25Okapi + whitespace tokens (original) | 0.23 | 0.32 | 0.32 | 0.32 | 5.0 | 0 |
| BM25Okapi + code tokens | 0.35 | 0.71 | 0.53 | 0.71 | 5.0 | 0 |
| BM25Plus + code tokens (current) | 0.32 | 0.69 | 0.51 | 0.69 | 5.0 | 0 |
| Dense (MiniLM + Qdrant) | 0.32 | 0.71 | 0.52 | 0.71 | 5.0 | 12 |
| Hybrid (RRF k=60) | 0.44 | 0.75 | 0.61 | 0.75 | 5.0 | 13 |
| Hybrid + cross-encoder rerank | 0.47 | 0.72 | 0.61 | 0.72 | 5.0 | 2674 |
| Hybrid + rerank + multi-hop | 0.47 | 0.72 | 0.61 | 0.82 | 13.1 | 2668 |

What it shows:

- The code-aware tokenizer more than doubled BM25's Recall@5 (0.32 → 0.71).
- Hybrid beats either retriever alone: Hit@1 0.44 vs 0.32, MRR 0.61 vs ~0.52.
- The general-purpose cross-encoder adds little here (Hit@1 +0.03, Recall@5 −0.03) and costs ~2.7 s per query on CPU. A code-trained reranker, or dropping it, is the next experiment.
- Multi-hop raises the share of gold chunks reaching the LLM from 0.72 to 0.82, at the cost of ~13 chunks of context instead of 5.
- n = 34 on one small repo, so differences of a few points are within noise.

`run_eval.py` judges the exact context the QA agent used and records unparseable judge replies as `null` (counted separately) rather than a default score.

## Tests

```bash
pytest
```

Unit tests cover the tokenizer, RRF, multi-hop limits, routing, diff queries, the sandbox, evaluation metrics, the chunker (qualified ids, 1-based lines, call extraction, decorators), and check that every benchmark gold chunk exists in the index.

## Known limitations

- Python only; the call graph resolves calls by bare name, not through imports.
- The embedder and reranker are general-purpose models, not code-specific.
- Re-indexing rebuilds everything; there is no incremental update yet.
- The test sandbox is process isolation, not a security boundary.

## Project layout

```
ingestion/   chunker.py (Tree-sitter), collect.py (file walk), graph_builder.py, indexer.py
retrieval/   tokenizer.py, fusion.py (RRF), hybrid_search.py, reranker.py, multi_hop.py
agents/      supervisor.py, routing.py, common.py, qa/review/test/docs agents, diff_query.py, sandbox.py
utils/       llm_factory.py (Gemini → Claude fallback)
api/         main.py (FastAPI)          public/   web UI
eval/        benchmark.json, metrics.py, run_ablation.py, run_eval.py
tests/       pytest suite              scripts/  manual smoke scripts that call the live LLMs
```
