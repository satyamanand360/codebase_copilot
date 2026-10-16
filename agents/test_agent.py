import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from agents.common import format_context, retrieve_context, strip_code_fences
from agents.sandbox import SandboxExecutor
from retrieval.hybrid_search import HybridSearcher
from retrieval.multi_hop import MultiHopExpander
from retrieval.reranker import Reranker
from utils.llm_factory import get_llm

load_dotenv()

MAX_ATTEMPTS = 3


class TestAgent:
    __test__ = False  # tell pytest this is not a test class

    def __init__(self, searcher=None, reranker=None, expander=None, sandbox=None):
        self.searcher = searcher or HybridSearcher()
        self.reranker = reranker or Reranker()
        self.expander = expander or MultiHopExpander()
        self.sandbox = sandbox or SandboxExecutor()
        self.run_tests = os.environ.get("ENABLE_TEST_SANDBOX") == "1"
        self.llm = get_llm("main")

        self.test_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert SDET. Write pytest unit tests for the target, covering happy paths and "
                       "edge cases, mocking external services. Import project code by its module path from the "
                       "repository root (for example `from retrieval.fusion import rrf_fuse`). "
                       "Output ONLY Python code."),
            ("user", "Context from codebase:\n{context}\n\nTarget:\n{target}{feedback}"),
        ])

    def generate_tests(self, target: str) -> str:
        chunks = retrieve_context(self.searcher, self.reranker, self.expander, target[:1000], top_k=20)
        if not self.llm:
            return "Error: Cannot generate tests. LLM API keys are not configured."

        chain = self.test_prompt | self.llm | StrOutputParser()
        context = format_context(chunks)
        feedback = ""
        code = ""
        attempts = MAX_ATTEMPTS if self.run_tests else 1
        for attempt in range(1, attempts + 1):
            code = strip_code_fences(chain.invoke({"context": context, "target": target, "feedback": feedback}))
            if not self.run_tests:
                return f"```python\n{code}```\n\n_Not executed. Set ENABLE_TEST_SANDBOX=1 to run generated tests._"
            result = self.sandbox.execute_test(code)
            if result["passed"]:
                return f"```python\n{code}```\n\n✅ Passed in sandbox on attempt {attempt}."
            # Give the model the pytest output and try again
            feedback = (f"\n\nYour previous tests failed when run with pytest.\nPrevious tests:\n{code}\n"
                        f"pytest output:\n{result['output'][-3000:]}\nFix the tests.")

        return (f"```python\n{code}```\n\n⚠️ Tests still failing after {attempts} attempts. "
                f"Last pytest output:\n```\n{result['output'][-1500:]}\n```")
