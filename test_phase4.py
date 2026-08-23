import os
from agents.supervisor import Supervisor

def main():
    supervisor = Supervisor()
    
    print("\n--- Phase 4 Test: Test Intent ---")
    test_input = "Generate a unit test for the add function in MathOperations"
    print(f"Input: {test_input}")
    
    result = supervisor.process(test_input)
    print(f"\nResult:\n{result}\n")

if __name__ == "__main__":
    main()
