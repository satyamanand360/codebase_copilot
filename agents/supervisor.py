from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph

from agents.docs_agent import DocsAgent
from agents.qa_agent import QAAgent
from agents.review_agent import ReviewAgent
from agents.routing import INTENTS, looks_like_diff, parse_intent
from agents.test_agent import TestAgent
from retrieval.hybrid_search import HybridSearcher
from retrieval.multi_hop import MultiHopExpander
from retrieval.reranker import Reranker
from utils.llm_factory import get_llm

load_dotenv()


class AppState(TypedDict):
    input_text: str
    intent: str
    final_output: str


class Supervisor:
    def __init__(self):
        # Load the expensive retrieval pieces once and share them across agents
        self.searcher = HybridSearcher()
        self.reranker = Reranker()
        self.expander = MultiHopExpander()

        shared = (self.searcher, self.reranker, self.expander)
        self.qa_agent = QAAgent(*shared)
        self.review_agent = ReviewAgent(*shared)
        self.test_agent = TestAgent(*shared)
        self.docs_agent = DocsAgent(*shared)

        self.llm = get_llm("fast")
        self.routing_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a routing supervisor. Classify the user's input into exactly one category: "
                       "'qa' (questions about the code), 'review' (code review or PR diffs), 'test' (write unit "
                       "tests) or 'docs' (write documentation). Output ONLY the category name in lowercase."),
            ("user", "{input_text}"),
        ])
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AppState)
        workflow.add_node("supervisor", self.classify_intent)
        workflow.add_node("qa", self.run_qa)
        workflow.add_node("review", self.run_review)
        workflow.add_node("test", self.run_test)
        workflow.add_node("docs", self.run_docs)
        workflow.set_entry_point("supervisor")
        workflow.add_conditional_edges("supervisor", lambda s: s["intent"], {i: i for i in INTENTS})
        for intent in INTENTS:
            workflow.add_edge(intent, END)
        return workflow.compile()

    # Nodes return only the keys they change; LangGraph merges them into state.
    def classify_intent(self, state: AppState):
        text = state["input_text"]
        if looks_like_diff(text):
            return {"intent": "review"}  # obvious case, no LLM call needed
        if not self.llm:
            return {"intent": "qa"}
        reply = (self.routing_prompt | self.llm | StrOutputParser()).invoke({"input_text": text})
        intent = parse_intent(reply)
        print(f"[Supervisor] router said {reply!r} -> {intent}")
        return {"intent": intent}

    def run_qa(self, state: AppState):
        return {"final_output": self.qa_agent.answer_question(state["input_text"])}

    def run_review(self, state: AppState):
        return {"final_output": self.review_agent.review_diff(state["input_text"])}

    def run_test(self, state: AppState):
        return {"final_output": self.test_agent.generate_tests(state["input_text"])}

    def run_docs(self, state: AppState):
        return {"final_output": self.docs_agent.generate_docs(state["input_text"])}

    def run(self, user_input: str) -> AppState:
        return self.graph.invoke({"input_text": user_input, "intent": "", "final_output": ""})

    def process(self, user_input: str) -> str:
        return self.run(user_input)["final_output"]
