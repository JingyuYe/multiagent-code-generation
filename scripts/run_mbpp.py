import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset
from src.graphs.g3_mapcoder import build_g3_mapcoder
from src.core.state import TaskState

def run_test():
    print("Loading a subset of MBPP (Sanitized) from HuggingFace...")
    # Grab the first two instances of the MBPP benchmark for quick testing
    dataset = load_dataset("mbpp", "sanitized", split="test[:2]")
    
    print("Initializing G3 MapCoder Cycle Graph using local Ollama model...")
    # Initialize the complex topology at low effort (0) to get fast evaluation times locally
    graph = build_g3_mapcoder(effort_level=0) 
    
    results = []
    
    for row in dataset:
        task_id = f"MBPP-{row['task_id']}"
        prompt = row['prompt']
        tests = row['test_list']
        
        task_input = {
            "task_id": task_id,
            "task_description": prompt,
            "tests": tests,
            "iterations": 0,
            "test_results": [],
            "review_comments": [],
            "token_usage": {},
            "code": "",
            "plan": "",
            "test_passed": False
        }
        
        print(f"\nEvaluating {task_id}...")
        print(f"Prompt: {prompt}")
        try:
            final_state = graph.invoke(task_input)
            pass_status = final_state.get('test_passed', False)
            print(f"[{task_id}] Completed. Pass: {pass_status}, Iterations: {final_state.get('iterations', 0)}")
            print(f"Tokens: {final_state.get('token_usage')}")
            results.append((task_id, pass_status))
            
            # Print last tester trace if failed
            if not pass_status and len(final_state.get('test_results', [])) > 0:
                print(f"Last Trace:\n{final_state.get('test_results')[-1]}")
                
        except Exception as e:
            print(f"[{task_id}] Failed during LangGraph execution: {e}")
            
    print("\n==== FINAL G3 RESULTS ====")
    passed = sum(1 for _, status in results if status)
    if len(results) > 0:
        print(f"Pass@1: {passed}/{len(results)} ({passed/len(results)*100:.1f}%)")

if __name__ == "__main__":
    run_test()
