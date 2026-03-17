import os
import sys
from rich.console import Console

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset
from src.core.state import TaskState

from src.graphs.g0_baseline import build_g0_baseline
from src.graphs.g1_waterfall import build_g1_waterfall
from src.graphs.g2_agentcoder import build_g2_agentcoder
from src.graphs.g3_mapcoder import build_g3_mapcoder
from src.graphs.g4_parallel_judge import build_g4_parallel_judge
from src.graphs.g5_adversarial_debate import build_g5_adversarial_debate
from src.graphs.g6_hierarchical import build_g6_hierarchical
from src.graphs.g7_mental_simulator import build_g7_mental_simulator

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
        "test_passed": False,
        "code_candidates": [],
        "debate_history": [],
        "subtasks": [],
        "subtask_code": [],
        "simulation_feedback": ""
    }

def main():
    console = Console()
    console.print("\n[bold cyan]🚀 Testing All Topologies on a Single Example[/bold cyan]")
    
    # Load 1 single task to test all configs
    dataset = load_dataset("mbpp", "sanitized", split="test[:1]")
    demo_task = dataset[0]
    task_id = f"MBPP-{demo_task['task_id']}"
    prompt = demo_task['prompt']
    tests = demo_task['test_list']
    
    console.print(f"\n[bold green]--- Testing Task: {task_id} ---[/bold green]")
    console.print(f"Prompt: {prompt}\n")
    
    topologies = [
        ("G0: Baseline", lambda: build_g0_baseline()),
        ("G1: Waterfall", lambda: build_g1_waterfall()),
        ("G2: AgentCoder", lambda: build_g2_agentcoder()),
        ("G3: MapCoder", lambda: build_g3_mapcoder()),
        ("G4: Parallel Judge", lambda: build_g4_parallel_judge()),
        ("G5: Adversarial Debate", lambda: build_g5_adversarial_debate()),
        ("G6: Hierarchical Setup", lambda: build_g6_hierarchical()),
        ("G7: Mental Simulator", lambda: build_g7_mental_simulator())
    ]
    
    for name, init_func in topologies:
        console.print(f"\n[bold yellow]Running {name}...[/bold yellow]")
        graph = init_func()
        try:
            state = graph.invoke(create_task_input(task_id, prompt, tests), config={"recursion_limit": 10})
            tokens = sum(state.get('token_usage', {}).values())
            passed = state.get('test_passed', False)
            console.print(f"✅ Finished. Passed: {passed} | Total Tokens: {tokens}")
        except Exception as e:
            console.print(f"❌ Failed: {e}")

if __name__ == "__main__":
    main()
