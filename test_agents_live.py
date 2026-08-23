import os
from agents.supervisor import Supervisor

def main():
    print("--- Starting Live Pipeline Test for Remaining Agents ---")
    supervisor = Supervisor()
    
    test_input = "Please write unit tests for the dummy_repo/string_utils.py file."
    print(f"\n--- Test 3: Test Intent ---")
    print(f"Input: {test_input}")
    test_result = supervisor.process(test_input)
    print(f"Result (Test): {test_result}\n")

    docs_input = "Draft docs for the string_utils.py file"
    print(f"\n--- Test 4: Docs Intent ---")
    print(f"Input: {docs_input}")
    docs_result = supervisor.process(docs_input)
    print(f"Result (Docs): {docs_result}\n")

if __name__ == "__main__":
    main()
