"""End-to-end LLM-as-judge evaluation (needs GOOGLE_API_KEY).

For each benchmark question the QA agent answers, and a judge LLM scores:
  Faithfulness       every claim in the answer is supported by the context
  Context precision  share of retrieved chunks that were needed
  Context recall     the context covers the ground-truth answer

The judge sees exactly the context the QA agent used. Unparseable judge
replies are recorded as null and counted, never replaced by a default.

    python eval/run_eval.py [--limit N] [--sleep SECONDS]
"""
import argparse
import json
import os
import sys
import time

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.common import format_context  # noqa: E402
from agents.qa_agent import QAAgent  # noqa: E402
from eval.metrics import mean, parse_score  # noqa: E402
from utils.llm_factory import get_llm  # noqa: E402

load_dotenv()
HERE = os.path.dirname(os.path.abspath(__file__))

RUBRIC = "Output ONLY a single number between 0.0 and 1.0, like 0.75."
FAITHFULNESS = ChatPromptTemplate.from_messages([
    ("system", "You are an evaluation judge scoring FAITHFULNESS: the fraction of claims in the ANSWER that are "
               "directly supported by the CONTEXT. 1.0 = every claim supported, 0.0 = none. " + RUBRIC),
    ("user", "CONTEXT:\n{context}\n\nANSWER:\n{answer}\n\nFaithfulness score:"),
])
PRECISION = ChatPromptTemplate.from_messages([
    ("system", "You are an evaluation judge scoring CONTEXT PRECISION: the fraction of the retrieved code chunks "
               "that are relevant to answering the QUESTION. " + RUBRIC),
    ("user", "QUESTION:\n{question}\n\nRETRIEVED CONTEXT:\n{context}\n\nContext precision score:"),
])
RECALL = ChatPromptTemplate.from_messages([
    ("system", "You are an evaluation judge scoring CONTEXT RECALL: how much of the GROUND TRUTH answer can be "
               "supported by the RETRIEVED CONTEXT. 1.0 = all of it, 0.0 = none. " + RUBRIC),
    ("user", "GROUND TRUTH:\n{ground_truth}\n\nRETRIEVED CONTEXT:\n{context}\n\nContext recall score:"),
])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only the first N questions")
    ap.add_argument("--sleep", type=float, default=4.0, help="seconds between LLM calls (free-tier rate limits)")
    args = ap.parse_args()

    judge = get_llm("fast")
    if judge is None:
        sys.exit("No LLM API key configured in .env")
    parser = StrOutputParser()
    chains = {"faithfulness": FAITHFULNESS | judge | parser,
              "context_precision": PRECISION | judge | parser,
              "context_recall": RECALL | judge | parser}

    with open(os.path.join(HERE, "benchmark.json"), encoding="utf-8") as f:
        bench = json.load(f)[: args.limit]

    agent = QAAgent()
    results = []
    for i, item in enumerate(bench, 1):
        q = item["question"]
        print(f"Q{i}/{len(bench)}: {q}")
        answer, chunks = agent.answer_with_context(q)
        ctx = format_context(chunks)
        row = {"question": q, "answer": answer, "context_ids": [c["chunk_id"] for c in chunks]}
        inputs = {"context": ctx, "answer": answer, "question": q, "ground_truth": item["ground_truth"]}
        for metric, chain in chains.items():
            time.sleep(args.sleep)
            raw = chain.invoke(inputs)
            row[metric] = parse_score(raw)
            if row[metric] is None:
                row[metric + "_raw"] = raw[:200]
        results.append(row)
        print("   " + "  ".join(f"{m}={row[m]}" for m in chains))

    aggregate = {}
    for m in chains:
        vals = [r[m] for r in results]
        aggregate[m] = {"mean": round(mean(vals), 3) if mean(vals) is not None else None,
                        "parsed": sum(v is not None for v in vals), "failed_to_parse": sum(v is None for v in vals)}
    out = {"n_questions": len(results), "aggregate": aggregate, "per_question": results}
    with open(os.path.join(HERE, "eval_results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
