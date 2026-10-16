"""Retrieval ablation on eval/benchmark.json. No LLM or API key needed.

    python -m ingestion.indexer .        # build the index first
    python eval/run_ablation.py

Each benchmark question lists the chunk ids that answer it ("gold_chunks").
For every retrieval configuration we measure:

  Hit@1        a gold chunk is ranked first
  Recall@5     share of gold chunks in the top 5
  MRR          mean of 1/rank of the first gold chunk (within the top 30)
  Ctx recall   share of gold chunks in the context the LLM would receive
               (top 5, plus graph neighbours for the multi-hop row)
  Ctx size     average number of chunks in that context

Results go to eval/ablation_results.json and eval/ablation_results.md.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rank_bm25 import BM25Okapi  # noqa: E402

from eval.metrics import hit_at_k, mean, recall_at_k, reciprocal_rank  # noqa: E402
from retrieval.hybrid_search import HybridSearcher  # noqa: E402
from retrieval.multi_hop import MultiHopExpander  # noqa: E402
from retrieval.reranker import Reranker  # noqa: E402
from retrieval.tokenizer import code_tokens  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TOP_K = 30
KEEP = 5


def main():
    with open(os.path.join(HERE, "benchmark.json"), encoding="utf-8") as f:
        bench = json.load(f)

    searcher = HybridSearcher()
    reranker = Reranker()
    expander = MultiHopExpander()

    if not searcher.chunks:
        sys.exit("Index is empty. Run `python -m ingestion.indexer .` first.")
    missing = sorted({g for q in bench for g in q["gold_chunks"] if g not in searcher.chunk_map})
    if missing:
        sys.exit("Benchmark refers to chunk ids that are not in the index:\n  " + "\n  ".join(missing))

    # The original setup (BM25Okapi + whitespace tokens) and an intermediate one
    # (BM25Okapi + code tokens), kept only to measure what each change did.
    old_bm25 = BM25Okapi([c["content"].lower().split() for c in searcher.chunks])
    okapi_code = BM25Okapi([code_tokens(c["content"]) for c in searcher.chunks])

    def ranked_ids(bm25, tokens):
        scores = bm25.get_scores(tokens)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:TOP_K]
        return [searcher.chunks[i]["chunk_id"] for i in order]

    def ids(chunks):
        return [c["chunk_id"] for c in chunks]

    configs = [
        ("BM25Okapi + whitespace tokens (original)", lambda q: (ranked_ids(old_bm25, q.lower().split()), None)),
        ("BM25Okapi + code tokens", lambda q: (ranked_ids(okapi_code, code_tokens(q)), None)),
        ("BM25Plus + code tokens (current)", lambda q: (ids(searcher.search(q, TOP_K, mode="bm25")), None)),
        ("Dense (MiniLM + Qdrant)", lambda q: (ids(searcher.search(q, TOP_K, mode="dense")), None)),
        ("Hybrid (RRF k=60)", lambda q: (ids(searcher.search(q, TOP_K)), None)),
    ]

    def rerank_cfg(q):
        ranked = reranker.rerank(q, searcher.search(q, TOP_K), top_k=TOP_K)
        return ids(ranked), None

    def multihop_cfg(q):
        ranked = reranker.rerank(q, searcher.search(q, TOP_K), top_k=TOP_K)
        context = expander.expand(ranked[:KEEP], searcher.chunks)
        return ids(ranked), ids(context)

    configs += [("Hybrid + cross-encoder rerank", rerank_cfg),
                ("Hybrid + rerank + multi-hop", multihop_cfg)]

    rows, per_question = [], []
    for name, fn in configs:
        start = time.time()
        stats = {"hit1": [], "r5": [], "mrr": [], "ctx_recall": [], "ctx_size": []}
        for q in bench:
            ranked, context = fn(q["question"])
            context = context or ranked[:KEEP]
            gold = q["gold_chunks"]
            stats["hit1"].append(hit_at_k(ranked, gold, 1))
            stats["r5"].append(recall_at_k(ranked, gold, 5))
            stats["mrr"].append(reciprocal_rank(ranked, gold))
            stats["ctx_recall"].append(recall_at_k(context, gold, len(context)))
            stats["ctx_size"].append(len(context))
            per_question.append({"config": name, "question": q["question"], "top5": ranked[:5],
                                 "gold": gold, "rr": stats["mrr"][-1]})
        row = {"config": name, **{k: round(mean(v), 3) for k, v in stats.items()},
               "ms_per_query": round(1000 * (time.time() - start) / len(bench))}
        rows.append(row)
        print(f"{name:40s} Hit@1={row['hit1']:.3f} R@5={row['r5']:.3f} MRR={row['mrr']:.3f} "
              f"CtxRecall={row['ctx_recall']:.3f} CtxSize={row['ctx_size']:.1f}")

    n_chunks = len(searcher.chunks)
    lines = [
        f"# Retrieval ablation ({len(bench)} questions, {n_chunks} indexed chunks)",
        "",
        "| Configuration | Hit@1 | Recall@5 | MRR | Context recall | Context size | ms/query |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['config']} | {r['hit1']:.2f} | {r['r5']:.2f} | {r['mrr']:.2f} | "
                     f"{r['ctx_recall']:.2f} | {r['ctx_size']:.1f} | {r['ms_per_query']} |")
    lines += ["", "Context = what the LLM receives: the top 5 chunks, plus call-graph neighbours in the last row.",
              "Latency is measured on whatever machine ran this (CPU on GitHub Actions)."]
    md = "\n".join(lines) + "\n"

    with open(os.path.join(HERE, "ablation_results.md"), "w", encoding="utf-8") as f:
        f.write(md)
    with open(os.path.join(HERE, "ablation_results.json"), "w", encoding="utf-8") as f:
        json.dump({"n_questions": len(bench), "n_chunks": n_chunks, "rows": rows,
                   "per_question": per_question}, f, indent=1)
    print("\n" + md)


if __name__ == "__main__":
    main()
