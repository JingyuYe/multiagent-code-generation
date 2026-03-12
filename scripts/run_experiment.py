import os
import sys

# Add project root to path so we can import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graphs.g2_agentcoder import build_g2_agentcoder
from src.game_theory.utility import calculate_utility

def run_test():
    print("Initializing G2 AgentCoder Graph using Qwen2.5-Coder:1.5B via Ollama...")
    graph = build_g2_agentcoder(effort_level=1)
    
    task_input = {
        "task_id": "MBPP-Test-1",
        "task_description": "Write a python function that takes a list of integers and returns the sum of all strictly positive integers.",
        "tests": [
            "assert sum_positive([1, -2, 3]) == 4",
            "assert sum_positive([-1, -2, -3]) == 0",
            "assert sum_positive([]) == 0"
        ],
        "iterations": 0,
        "test_results": [],
        "review_comments": [],
        "token_usage": {}
    }
    
    print("Executing Task...")
    # NOTE: In a real run, you want to parse the final state
    result_state = None
    try:
        # invoke returns the final dictionary state
        # In newer langgraph usage: graph.invoke(task_input) 
        pass
    except Exception as e:
         print(f"Exception during LLM call: Make sure Ollama is running and model qwen2.5-coder:1.5b is pulled. {e}")
         return
         
    print(f"Graph execution complete.")
    

if __name__ == "__main__":
    print("==== Local Testing Entrypoint ====")
    run_test()
