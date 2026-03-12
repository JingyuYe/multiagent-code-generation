import os
import sys
import json
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset
from src.graphs.g3_mapcoder import build_g3_mapcoder
from src.core.state import TaskState
from src.game_theory.utility import calculate_utility
from src.game_theory.stability import is_nash_equilibrium

def log_result(log_file, data):
    with open(log_file, 'a') as f:
        f.write(json.dumps(data) + "\n")

def create_task_input(task_id, prompt, tests):
    return {
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

def run_test():
    print("Loading a subset of MBPP (Sanitized) from HuggingFace...")
    # INCREASE BATCH SIZE: Running 5 tasks for the sweep
    dataset = load_dataset("mbpp", "sanitized", split="test[:5]")
    
    log_dir = Path("data")
    log_dir.mkdir(exist_ok=True)
    results_file = log_dir / "mbpp_results.jsonl"
    
    # Clear old results if starting fresh
    if results_file.exists():
        results_file.unlink()
        
    print(f"Results will be logged to {results_file}")
    
    roles = ["planner", "coder", "reviewer", "tester"]
    overall_results = []
    
    for row in tqdm(dataset, desc="Evaluating MBPP", disable=True): # disable tqdm bar to print logs cleanly
        task_id = f"MBPP-{row['task_id']}"
        prompt = row['prompt']
        tests = row['test_list']
        
        print(f"\n--- [BASE RUN: High Effort] {task_id} ---")
        base_profile = {r: 1 for r in roles}
        graph = build_g3_mapcoder(effort_profile=base_profile)
        
        try:
            base_state = graph.invoke(create_task_input(task_id, prompt, tests))
            base_pass = base_state.get('test_passed', False)
            base_tokens = base_state.get('token_usage', {})
            base_utils = calculate_utility(base_pass, base_tokens, value_of_success=100.0, lambda_cost=0.01)
            
            print(f"Base Pass: {base_pass}, Tokens: {base_tokens}")
            print(f"Base Utilities: {base_utils}")
            
            # --- Deviation Runs for Nash Equilibrium ---
            deviation_utils = {}
            for agent in roles:
                print(f"  > [DEVIATION: {agent} e=0] {task_id}")
                dev_profile = base_profile.copy()
                dev_profile[agent] = 0
                
                graph_dev = build_g3_mapcoder(effort_profile=dev_profile)
                dev_state = graph_dev.invoke(create_task_input(task_id, prompt, tests))
                
                dev_pass = dev_state.get('test_passed', False)
                dev_tokens = dev_state.get('token_usage', {})
                # calculate utility under deviation
                dev_u = calculate_utility(dev_pass, dev_tokens, value_of_success=100.0, lambda_cost=0.01)
                
                # record only the deviating agent's new utility to compare against base
                deviation_utils[agent] = dev_u.get(agent, 0.0)
                
            # Check Incentive Stability
            stability = is_nash_equilibrium(base_utils, deviation_utils, epsilon=5.0) # Epsilon tolerance
            
            print(f"> Stability Profile [{task_id}]: {stability}")
            
            log_data = {
                "task_id": task_id,
                "base_passed": base_pass,
                "base_tokens": base_tokens,
                "base_utilities": base_utils,
                "deviation_utilities": deviation_utils,
                "nash_equilibrium_stable": stability
            }
            
            log_result(results_file, log_data)
            overall_results.append(log_data)
            
        except Exception as e:
            print(f"[{task_id}] Failed during execution: {e}")
            
    # Print summary
    if overall_results:
        passed = sum(1 for r in overall_results if r["base_passed"])
        print(f"\n==== FINAL G3 RESULTS ====")
        print(f"Tasks Evaluated: {len(overall_results)}")
        print(f"Pass@1 (Base): {passed}/{len(overall_results)} ({passed/len(overall_results)*100:.1f}%)")
        
        # Stability aggregates
        for agent in roles:
            stable = sum(1 for r in overall_results if r["nash_equilibrium_stable"].get(agent, False))
            print(f"Agent '{agent}' strictly incentivized to maintain high effort: {stable}/{len(overall_results)} tasks")

if __name__ == "__main__":
    run_test()
