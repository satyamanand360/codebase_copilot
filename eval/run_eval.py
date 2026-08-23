"""
Custom LLM-as-Judge Evaluator for Codebase Copilot.
Computes Faithfulness, Context Precision, and Context Recall
using Google Gemini as the judge — no ragas dependency required.

Scoring rubric (each metric scored 0.0 to 1.0):
  Faithfulness      — Is every claim in the answer grounded in the context?
  Context Precision — What fraction of retrieved chunks were actually needed?
  Context Recall    — Did we retrieve all the necessary information?
"""

import os
import json
import sys
import time
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from retrieval.hybrid_search import HybridSearcher
from retrieval.reranker import Reranker
from retrieval.multi_hop import MultiHopExpander
from agents.qa_agent import QAAgent

load_dotenv()

FAITHFULNESS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an evaluation judge. Your task is to score FAITHFULNESS.
Faithfulness measures whether every claim in the ANSWER is fully supported by the CONTEXT.
Score from 0.0 to 1.0:
  1.0 = Every single statement in the answer is directly supported by the context.
  0.5 = Some statements are supported, some are not or are inferred.
  0.0 = The answer contains claims not present in the context at all (hallucinations).
Output ONLY a single float number like: 0.85"""),
    ("user", "CONTEXT:\n{context}\n\nANSWER:\n{answer}\n\nFaithfulness score:")
])

PRECISION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an evaluation judge. Your task is to score CONTEXT PRECISION.
Context Precision measures how many of the retrieved context chunks were actually relevant/needed to answer the question.
Score from 0.0 to 1.0:
  1.0 = All retrieved chunks were relevant and needed.
  0.5 = About half the chunks were relevant, rest were noise.
  0.0 = None of the retrieved chunks were relevant to the question.
Output ONLY a single float number like: 0.75"""),
    ("user", "QUESTION:\n{question}\n\nRETRIEVED CONTEXT CHUNKS:\n{context}\n\nContext Precision score:")
])

RECALL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an evaluation judge. Your task is to score CONTEXT RECALL.
Context Recall measures whether the retrieved context contained ALL the information needed to produce the ground truth answer.
Score from 0.0 to 1.0:
  1.0 = The context fully covers everything stated in the ground truth.
  0.5 = The context partially covers the ground truth.
  0.0 = The context is missing key information from the ground truth.
Output ONLY a single float number like: 0.90"""),
    ("user", "GROUND TRUTH ANSWER:\n{ground_truth}\n\nRETRIEVED CONTEXT:\n{context}\n\nContext Recall score:")
])

def safe_float(s):
    try:
        return float(s.strip().split()[0])
    except:
        return 0.5

def run_evaluation():
    print("=" * 65)
    print("  CODEBASE COPILOT — LLM-as-Judge Evaluation")
    print("  Metrics: Faithfulness | Context Precision | Context Recall")
    print("=" * 65)

    google_key = os.environ.get("GOOGLE_API_KEY")
    if not google_key or google_key == "your_google_api_key_here":
        print("Error: Missing GOOGLE_API_KEY in .env")
        return

    judge = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0, api_key=google_key)
    parser = StrOutputParser()

    faith_chain     = FAITHFULNESS_PROMPT | judge | parser
    precision_chain = PRECISION_PROMPT    | judge | parser
    recall_chain    = RECALL_PROMPT       | judge | parser

    with open("eval/benchmark.json", "r") as f:
        data = json.load(f)

    print(f"\n[Setup] Loaded {len(data)} benchmark questions.")
    print("[Setup] Initialising retrieval pipeline...\n")

    searcher  = HybridSearcher()
    reranker  = Reranker()
    expander  = MultiHopExpander()
    qa_agent  = QAAgent(searcher, reranker, expander)

    results = []

    for i, item in enumerate(data):
        q  = item["question"]
        gt = item["ground_truth"]

        print(f"Q{i+1}: {q}")

        # --- Retrieval ---
        retrieved   = searcher.search(q, top_k=20)
        reranked    = reranker.rerank(q, retrieved, top_k=5)
        expanded    = expander.expand(reranked, searcher.chunks)
        ctx_chunks  = [c["content"] for c in expanded]
        ctx_str     = "\n\n---\n\n".join(ctx_chunks)

        # --- Generation ---
        answer = qa_agent.answer_question(q)
        time.sleep(15)  # respect 5 RPM free-tier limit
        print(f"     Answer: {answer[:120]}...")

        # --- Judge ---
        f_score = safe_float(faith_chain.invoke({"context": ctx_str, "answer": answer}))
        time.sleep(15)
        p_score = safe_float(precision_chain.invoke({"question": q, "context": ctx_str}))
        time.sleep(15)
        r_score = safe_float(recall_chain.invoke({"ground_truth": gt, "context": ctx_str}))
        time.sleep(15)

        results.append({"question": q, "faithfulness": f_score,
                         "context_precision": p_score, "context_recall": r_score})

        print(f"     Faithfulness={f_score:.2f}  Precision={p_score:.2f}  Recall={r_score:.2f}\n")

    # --- Aggregates ---
    avg_f = sum(r["faithfulness"]       for r in results) / len(results)
    avg_p = sum(r["context_precision"]  for r in results) / len(results)
    avg_r = sum(r["context_recall"]     for r in results) / len(results)

    print("=" * 65)
    print("  AGGREGATE EVALUATION RESULTS")
    print("=" * 65)
    print(f"  Faithfulness      (anti-hallucination) : {avg_f:.3f} / 1.000")
    print(f"  Context Precision (retrieval accuracy) : {avg_p:.3f} / 1.000")
    print(f"  Context Recall    (retrieval coverage) : {avg_r:.3f} / 1.000")
    print("=" * 65)
    print("\nUse these numbers in your resume!")
    print('Example bullet: "Achieved {:.0f}% Faithfulness and {:.0f}% Context Precision'.format(avg_f*100, avg_p*100))
    print(' on a 5-question RAGAS-style benchmark evaluated by Gemini-as-judge."')

    # Save results to JSON
    with open("eval/eval_results.json", "w") as f:
        json.dump({
            "per_question": results,
            "aggregate": {
                "faithfulness": round(avg_f, 3),
                "context_precision": round(avg_p, 3),
                "context_recall": round(avg_r, 3)
            }
        }, f, indent=2)
    print("\nFull results saved to eval/eval_results.json")

if __name__ == "__main__":
    run_evaluation()
