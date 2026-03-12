import os
import sys

# Add project root to path so we can import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graphs.g0_baseline import build_g0_baseline

def run_test():
    print("Initializing G0 Baseline Graph using Qwen2.5-Coder:1.5B via Ollama...")
    graph = build_g0_baseline(effort_level=0) # Low effort for fast test
    
    task_input = {
        "task_id": "MBPP-Test-2",
        "task_description": "Write a python function `sum_positive(l)` that takes a list of integers and returns the sum of all strictly positive integers.",
        "tests": [
            "assert sum_positive([1, -2, 3]) == 4",
            "assert sum_positive([-1, -2, -3]) == 0",
            "assert sum_positive([]) == 0"
        ],
        "iterations": 0,
        "test_results": [],
        "review_comments": [],
        "token_usage": {},
        "code": "",
        "plan": "",
        "test_passed": False
    }
    
    print("Executing Task...")
    try:
        final_state = graph.invoke(task_input)
        
        print("\n==== EXECUTION COMPLETE ====")
        print(f"Token Usage: {final_state.get('token_usage', {})}")
        print(f"Generated Code:\n{final_state.get('code', 'N/A')}")
        print(f"\nTest Passed: {final_state.get('test_passed', False)}")
        print(f"Test Details:\n{final_state.get('test_results', [])[-1] if final_state.get('test_results') else 'N/A'}")
        
    except Exception as e:
         print(f"Exception during execution: {e}")
         print("Make sure Ollama is running and qwen2.5-coder:1.5b is pulled:")
         print("  ollama run qwen2.5-coder:1.5b")
         

if __name__ == "__main__":
    print("==== Local Testing Entrypoint (G0) ====")
    run_test()
