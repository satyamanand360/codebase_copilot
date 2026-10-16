from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from agents.common import format_context, retrieve_context
from retrieval.hybrid_search import HybridSearcher
from retrieval.multi_hop import MultiHopExpander
from retrieval.reranker import Reranker
from utils.llm_factory import get_llm

load_dotenv()


class DocsAgent:
    def __init__(self, searcher=None, reranker=None, expander=None):
        self.searcher = searcher or HybridSearcher()
        self.reranker = reranker or Reranker()
        self.expander = expander or MultiHopExpander()
        self.llm = get_llm("main")

        self.docs_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert technical writer. You are given a request or a piece of code and related "
                       "code from the codebase. Write Markdown documentation with parameter descriptions and usage "
                       "examples. Only document behaviour that the code shows."),
            ("user", "Context from codebase:\n{context}\n\nTarget:\n{target}\n\nPlease generate the documentation."),
        ])

    def generate_docs(self, target: str) -> str:
        chunks = retrieve_context(self.searcher, self.reranker, self.expander, target[:1000], top_k=20)
        if not self.llm:
            return "Error: Cannot generate docs. LLM API keys are not configured."
        chain = self.docs_prompt | self.llm | StrOutputParser()
        return chain.invoke({"context": format_context(chunks), "target": target})
