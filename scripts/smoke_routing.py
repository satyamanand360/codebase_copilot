import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.supervisor import Supervisor

def main():
    supervisor = Supervisor()
    
    print("\n--- Test 1: QA Intent ---")
    qa_input = "What does the multiply function do?"
    print(f"Input: {qa_input}")
    qa_result = supervisor.process(qa_input)
    print(f"Result (QA): {qa_result[:100]}...\n")
    
    print("\n--- Test 2: Review Intent ---")
    review_input = "Please review this PR diff:\n+ def multiply(a, b):\n+     return a * b"
    print(f"Input: {review_input}")
    review_result = supervisor.process(review_input)
    print(f"Result (Review): {review_result[:100]}...\n")

if __name__ == "__main__":
    main()
