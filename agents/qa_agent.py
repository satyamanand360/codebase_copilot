from utils.llm_factory import get_llm
from langchain_core.output_parsers import StrOutputParser
import os
from typing import List, Dict
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

from retrieval.hybrid_search import HybridSearcher
from retrieval.reranker import Reranker
from retrieval.multi_hop import MultiHopExpander

load_dotenv()

class QAAgent:
    def __init__(self, searcher=None, reranker=None, expander=None):
        self.searcher = searcher or HybridSearcher()
        self.reranker = reranker or Reranker()
        self.expander = expander or MultiHopExpander()
        
        self.llm = get_llm("main")
        self.verifier_llm = get_llm("fast")

        self.generation_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert codebase assistant. Answer the user's question using ONLY the provided code chunks. You MUST cite the file path and exact line numbers you used to formulate your answer."),
            ("user", "Context:\n{context}\n\nQuestion: {question}")
        ])
        
        self.verification_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an accuracy verification assistant. Check if the 'Answer' is fully supported by the 'Context'. If it contains hallucinations or unsupported claims, output 'FAIL: <reason>'. If it is fully supported, output 'PASS'."),
            ("user", "Context:\n{context}\n\nAnswer to verify:\n{answer}")
        ])

    def format_context(self, chunks: List[Dict]) -> str:
        formatted = []
        for c in chunks:
            formatted.append(f"File: {c['file_path']} (Lines {c['start_line']}-{c['end_line']})\nSymbol: {c['symbol_name']}\nCode:\n```\n{c['content']}\n```")
        return "\n\n".join(formatted)

    def answer_question(self, question: str) -> str:
        print(f"Question: {question}")
        print("1. Performing Hybrid Search...")
        top_k_retrieved = self.searcher.search(question, top_k=30)
        
        print("2. Reranking results...")
        top_5_reranked = self.reranker.rerank(question, top_k_retrieved, top_k=5)
        
        print("3. Expanding via Dependency Graph...")
        expanded_context = self.expander.expand(top_5_reranked, self.searcher.chunks)
        
        print(f"   -> Gathered {len(expanded_context)} code chunks for context.")
        context_str = self.format_context(expanded_context)
        
        if not self.llm:
            return "Error: Cannot generate answer. LLM API keys are not configured."
            
        for attempt in range(3):
            print(f"4. Generating Answer (Attempt {attempt+1}/3)...")
            chain = self.generation_prompt | self.llm | StrOutputParser()
            answer = chain.invoke({"context": context_str, "question": question})
            
            print("5. Verifying Answer...")
            verify_chain = self.verification_prompt | self.verifier_llm | StrOutputParser()
            verification = verify_chain.invoke({"context": context_str, "answer": answer})
            
            if verification.strip().startswith("PASS"):
                print("   -> Verification Passed!")
                return answer
            else:
                print(f"   -> Verification Failed: {verification}")
                
        return f"Failed to generate a verifiable answer after 3 attempts. Last attempted answer:\n{answer}"
