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

class DocsAgent:
    def __init__(self, searcher=None, reranker=None, expander=None):
        self.searcher = searcher or HybridSearcher()
        self.reranker = reranker or Reranker()
        self.expander = expander or MultiHopExpander()
        
        self.llm = get_llm("main")
            
        self.docs_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert technical writer. You are given a piece of code (like a class or file) and the surrounding context of the codebase. Generate comprehensive, markdown-formatted documentation for it, including usage examples and parameter descriptions."),
            ("user", "Context from Codebase:\n{context}\n\nTarget Code to Document:\n{target}\n\nPlease generate the documentation.")
        ])

    def format_context(self, chunks: List[Dict]) -> str:
        formatted = []
        for c in chunks:
            formatted.append(f"File: {c['file_path']} (Lines {c['start_line']}-{c['end_line']})\nSymbol: {c['symbol_name']}\nCode:\n```\n{c['content']}\n```")
        return "\n\n".join(formatted)

    def generate_docs(self, target_code: str) -> str:
        print("1. Docs Agent extracting keywords from target code...")
        query = target_code[:500]
        
        print("2. Retrieving Context...")
        top_k_retrieved = self.searcher.search(query, top_k=20)
        top_5_reranked = self.reranker.rerank(query, top_k_retrieved, top_k=5)
        
        print("3. Expanding Context for Callers/Callees...")
        expanded_context = self.expander.expand(top_5_reranked, self.searcher.chunks)
        
        context_str = self.format_context(expanded_context)
        
        if not self.llm:
            return "Error: Cannot generate docs. LLM API keys are not configured."

        print("4. Generating Documentation...")
        chain = self.docs_prompt | self.llm | StrOutputParser()
        docs = chain.invoke({"context": context_str, "target": target_code})
        return docs
