import os
from typing import TypedDict, Annotated, Sequence
import operator
from utils.llm_factory import get_llm
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate

from retrieval.hybrid_search import HybridSearcher
from retrieval.reranker import Reranker
from retrieval.multi_hop import MultiHopExpander

from agents.qa_agent import QAAgent
from agents.review_agent import ReviewAgent
from agents.test_agent import TestAgent
from agents.docs_agent import DocsAgent

load_dotenv()

class AppState(TypedDict):
    input_text: str
    intent: str
    final_output: str

class Supervisor:
    def __init__(self):
        # 1. Initialize Retrieval Components Once
        self.searcher = HybridSearcher()
        self.reranker = Reranker()
        self.expander = MultiHopExpander()
        
        # 2. Inject them into Agents
        self.qa_agent = QAAgent(self.searcher, self.reranker, self.expander)
        self.review_agent = ReviewAgent(self.searcher, self.reranker, self.expander)
        self.test_agent = TestAgent(self.searcher, self.reranker, self.expander)
        self.docs_agent = DocsAgent(self.searcher, self.reranker, self.expander)
        
        # 3. Setup fast LLM for intent routing
        self.llm = get_llm("fast")

        self.routing_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a routing supervisor. Given the user's input, classify their intent into exactly one of these categories: 'qa' (general questions), 'review' (code review or PR diffs), 'test' (generating unit tests), or 'docs' (generating documentation). Output ONLY the category name in lowercase."),
            ("user", "{input_text}")
        ])

        # 4. Build LangGraph
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AppState)
        
        workflow.add_node("supervisor", self.classify_intent)
        workflow.add_node("qa", self.run_qa)
        workflow.add_node("review", self.run_review)
        workflow.add_node("test", self.run_test)
        workflow.add_node("docs", self.run_docs)
        
        workflow.set_entry_point("supervisor")
        
        workflow.add_conditional_edges(
            "supervisor",
            lambda state: state["intent"],
            {
                "qa": "qa",
                "review": "review",
                "test": "test",
                "docs": "docs"
            }
        )
        
        workflow.add_edge("qa", END)
        workflow.add_edge("review", END)
        workflow.add_edge("test", END)
        workflow.add_edge("docs", END)
        
        return workflow.compile()

    def classify_intent(self, state: AppState):
        print("[Supervisor] Classifying intent...")
        if not self.llm:
            return {"intent": "qa"} # fallback default
            
        chain = self.routing_prompt | self.llm | StrOutputParser()
        intent = chain.invoke({"input_text": state["input_text"]}).strip().lower()
        
        if intent not in ["qa", "review", "test", "docs"]:
            intent = "qa"
            
        print(f"[Supervisor] Routing to {intent.capitalize()}Agent...")
        return {"intent": intent}

    def run_qa(self, state: AppState):
        state["final_output"] = self.qa_agent.answer_question(state["input_text"])
        return state

    def run_review(self, state: AppState):
        state["final_output"] = self.review_agent.review_diff(state["input_text"])
        return state
        
    def run_test(self, state: AppState):
        state["final_output"] = self.test_agent.generate_tests(state["input_text"])
        return state
        
    def run_docs(self, state: AppState):
        state["final_output"] = self.docs_agent.generate_docs(state["input_text"])
        return state

    def process(self, user_input: str) -> str:
        initial_state = {"input_text": user_input, "intent": "", "final_output": ""}
        result_state = self.graph.invoke(initial_state)
        return result_state["final_output"]
