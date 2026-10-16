import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.qa_agent import QAAgent

def main():
    agent = QAAgent()
    
    question = "How does the multiply function work in MathOperations, and what is its relation to the add function?"
    print("\n--- Starting Q&A Pipeline ---\n")
    
    # We expect Anthropic API key to be set. If not, it will fail gracefully in the agent.
    answer = agent.answer_question(question)
    
    print("\n--- Final Answer ---\n")
    print(answer)

if __name__ == "__main__":
    main()
