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

class ReviewAgent:
    def __init__(self, searcher=None, reranker=None, expander=None):
        self.searcher = searcher or HybridSearcher()
        self.reranker = reranker or Reranker()
        self.expander = expander or MultiHopExpander()
        
        self.llm = get_llm("main")
            
        self.review_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert code reviewer. You are given a unified diff of a pull request, along with context from the codebase. Provide a detailed, constructive critique on the style, correctness, and potential architectural impacts (e.g. breaking callers)."),
            ("user", "Context from Codebase:\n{context}\n\nPull Request Diff:\n{diff}\n\nPlease provide your review.")
        ])

    def format_context(self, chunks: List[Dict]) -> str:
        formatted = []
        for c in chunks:
            formatted.append(f"File: {c['file_path']} (Lines {c['start_line']}-{c['end_line']})\nSymbol: {c['symbol_name']}\nCode:\n```\n{c['content']}\n```")
        return "\n\n".join(formatted)

    def review_diff(self, diff_content: str) -> str:
        print("1. Review Agent extracting keywords from diff...")
        query = diff_content[:500] 
        
        print("2. Retrieving Context...")
        top_k_retrieved = self.searcher.search(query, top_k=20)
        top_5_reranked = self.reranker.rerank(query, top_k_retrieved, top_k=5)
        
        print("3. Expanding Context for Callers/Callees...")
        expanded_context = self.expander.expand(top_5_reranked, self.searcher.chunks)
        
        context_str = self.format_context(expanded_context)
        
        if not self.llm:
            return "Error: Cannot generate review. LLM API keys are not configured."

        print("4. Generating Code Review...")
        chain = self.review_prompt | self.llm | StrOutputParser()
        review = chain.invoke({"context": context_str, "diff": diff_content})
        return review
