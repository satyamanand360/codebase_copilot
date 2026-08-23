import os
from agents.supervisor import Supervisor

def main():
    print("--- Phase 5 Test: Docs Intent ---")
    supervisor = Supervisor()
    
    docs_input = "Draft docs for the string_utils.py file"
    print(f"Input: {docs_input}")
    
    result = supervisor.process(docs_input)
    print(f"\nResult:\n{result}\n")
    
    print("--- Phase 6 Test: Evaluation Script ---")
    print("Running run_eval.py as a subprocess to verify it imports correctly.")
    os.system("python eval/run_eval.py")

if __name__ == "__main__":
    main()
