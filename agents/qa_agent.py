from typing import Dict, List, Tuple

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from agents.common import format_context, retrieve_context
from retrieval.hybrid_search import HybridSearcher
from retrieval.multi_hop import MultiHopExpander
from retrieval.reranker import Reranker
from utils.llm_factory import get_llm

load_dotenv()

MAX_ATTEMPTS = 3


class QAAgent:
    def __init__(self, searcher=None, reranker=None, expander=None):
        self.searcher = searcher or HybridSearcher()
        self.reranker = reranker or Reranker()
        self.expander = expander or MultiHopExpander()

        self.llm = get_llm("main")
        self.verifier_llm = get_llm("fast")

        self.generation_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert codebase assistant. Answer the user's question using ONLY the provided "
                       "code chunks. Cite the file path and line numbers you used, in the form path:start-end. "
                       "If the chunks do not contain the answer, say so instead of guessing."),
            ("user", "Context:\n{context}\n\nQuestion: {question}{feedback}"),
        ])
        self.verification_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an accuracy verification assistant. Check if the 'Answer' is fully supported by the "
                       "'Context'. If it contains hallucinations or unsupported claims, output 'FAIL: <reason>'. "
                       "If it is fully supported, output 'PASS'."),
            ("user", "Context:\n{context}\n\nAnswer to verify:\n{answer}"),
        ])

    def answer_with_context(self, question: str) -> Tuple[str, List[Dict]]:
        """Returns (answer, context_chunks) so evaluation can judge the exact context used."""
        chunks = retrieve_context(self.searcher, self.reranker, self.expander, question, top_k=30)
        print(f"   -> {len(chunks)} code chunks in context")
        context = format_context(chunks)

        if not self.llm:
            return "Error: Cannot generate answer. LLM API keys are not configured.", chunks

        generate = self.generation_prompt | self.llm | StrOutputParser()
        verify = self.verification_prompt | (self.verifier_llm or self.llm) | StrOutputParser()

        feedback = ""
        answer = ""
        for attempt in range(1, MAX_ATTEMPTS + 1):
            answer = generate.invoke({"context": context, "question": question, "feedback": feedback})
            verdict = verify.invoke({"context": context, "answer": answer}).strip()
            if verdict.upper().startswith("PASS"):
                print(f"   -> verified on attempt {attempt}")
                return answer, chunks
            print(f"   -> attempt {attempt} rejected: {verdict}")
            # Feed the rejection back; re-sending the same prompt at temperature 0
            # would just reproduce the same answer.
            feedback = (
                f"\n\nA reviewer rejected your previous answer.\nPrevious answer:\n{answer}\n"
                f"Reason: {verdict}\nWrite a corrected answer that only states what the context supports."
            )

        return f"(Could not fully verify this answer after {MAX_ATTEMPTS} attempts.)\n\n{answer}", chunks

    def answer_question(self, question: str) -> str:
        return self.answer_with_context(question)[0]
