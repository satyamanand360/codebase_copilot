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

class TestAgent:
    def __init__(self, searcher=None, reranker=None, expander=None):
        self.searcher = searcher or HybridSearcher()
        self.reranker = reranker or Reranker()
        self.expander = expander or MultiHopExpander()
        
        self.llm = get_llm("main")
            
        self.test_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert SDET (Software Development Engineer in Test). You are given a source code snippet and the surrounding context of the codebase. Generate robust unit tests (e.g., using pytest for Python) covering edge cases, happy paths, and mocking dependencies where appropriate. Output ONLY the test code."),
            ("user", "Context from Codebase:\n{context}\n\nTarget Code to Test:\n{target}\n\nPlease generate unit tests.")
        ])

    def format_context(self, chunks: List[Dict]) -> str:
        formatted = []
        for c in chunks:
            formatted.append(f"File: {c['file_path']} (Lines {c['start_line']}-{c['end_line']})\nSymbol: {c['symbol_name']}\nCode:\n```\n{c['content']}\n```")
        return "\n\n".join(formatted)

    def generate_tests(self, target_code: str) -> str:
        print("1. Test Agent extracting keywords from target code...")
        query = target_code[:500]
        
        print("2. Retrieving Context...")
        top_k_retrieved = self.searcher.search(query, top_k=20)
        top_5_reranked = self.reranker.rerank(query, top_k_retrieved, top_k=5)
        
        print("3. Expanding Context for Callers/Callees...")
        expanded_context = self.expander.expand(top_5_reranked, self.searcher.chunks)
        
        context_str = self.format_context(expanded_context)
        
        if not self.llm:
            return "Error: Cannot generate tests. LLM API keys are not configured."

        print("4. Generating Unit Tests...")
        chain = self.test_prompt | self.llm | StrOutputParser()
        tests = chain.invoke({"context": context_str, "target": target_code})
        return tests
