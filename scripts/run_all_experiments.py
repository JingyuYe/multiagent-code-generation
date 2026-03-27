import os
import sys
import json
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset
from src.graphs.g0_baseline import build_g0_baseline
from src.graphs.g1_waterfall import build_g1_waterfall
from src.graphs.g2_agentcoder import build_g2_agentcoder
from src.graphs.g3_mapcoder import build_g3_mapcoder
from src.graphs.g4_parallel_judge import build_g4_parallel_judge
from src.graphs.g5_adversarial_debate import build_g5_adversarial_debate
from src.graphs.g6_hierarchical import build_g6_hierarchical
from src.graphs.g7_mental_simulator import build_g7_mental_simulator
from src.core.state import TaskState
from src.game_theory.utility import calculate_utility
from src.game_theory.stability import is_nash_equilibrium

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

def load_existing_results():
    """Reads raw_results.jsonl and returns a nested dict {topology: {task_id: result_obj}}."""
    results = {}
    raw_results_file = Path("data/raw_results.jsonl")
    if raw_results_file.exists():
        with open(raw_results_file, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    topology = data.get("topology")
                    task_id = data.get("task_id")
                    if topology and task_id:
                        if topology not in results:
                            results[topology] = {}
                        results[topology][task_id] = data
                except json.JSONDecodeError:
                    continue
    return results

def evaluate_task(row, builder_func, graph_name, roles, base_profile, console):
    """
    Evaluates a single task: base run + all deviation runs.
    """
    task_id = f"MBPP-{row['task_id']}"
    prompt = row['prompt']
    tests = row['test_list']
    
    try:
        # Base Run
        if graph_name == "G0: Baseline (0-shot)":
            graph = builder_func(effort_level=1)
            # console.print(f"  [cyan]Task {task_id}: Base Run (universal_agent: 1)[/cyan]")
        else:
            graph = builder_func(effort_profile=base_profile)
            # console.print(f"  [cyan]Task {task_id}: Base Run {base_profile}[/cyan]")
        
        # INCREASED RECURSION LIMIT TO 50
        base_state = graph.invoke(create_task_input(task_id, prompt, tests), config={"recursion_limit": 50})
        base_pass = base_state.get('test_passed', False)
        base_tokens = base_state.get('token_usage', {})
        base_utils = calculate_utility(base_pass, base_tokens, value_of_success=100.0, lambda_cost=0.01)
        
        stability = {}
        # Check deviations only if there are specialized roles.
        if graph_name != "G0: Baseline (0-shot)":
            deviation_utils = {}
            for agent in roles:
                dev_profile = base_profile.copy()
                dev_profile[agent] = 0
                # console.print(f"  [magenta]Task {task_id}: Deviation Run {dev_profile}[/magenta]")
                graph_dev = builder_func(effort_profile=dev_profile)
                dev_state = graph_dev.invoke(create_task_input(task_id, prompt, tests), config={"recursion_limit": 50})
                dev_pass = dev_state.get('test_passed', False)
                dev_tokens = dev_state.get('token_usage', {})
                dev_u = calculate_utility(dev_pass, dev_tokens, value_of_success=100.0, lambda_cost=0.01)
                deviation_utils[agent] = dev_u.get(agent, 0.0)
                
            stability = is_nash_equilibrium(base_utils, deviation_utils, epsilon=5.0)
        else:
            stability = {"universal_agent": True} # Default stable for baseline
        
        result_obj = {
            "topology": graph_name,
            "task_id": task_id,
            "passed": base_pass,
            "tokens": sum(base_tokens.values()),
            "stability": stability
        }
        
        # Stream result to disk safely (appended)
        raw_results_file = Path("data/raw_results.jsonl")
        with open(raw_results_file, "a") as f:
            f.write(json.dumps(result_obj) + "\n")
            
        return result_obj
        
    except Exception as e:
        console.print(f"[red]Error on {task_id}: {e}[/red]")
        return None

def run_all_experiments():
    console = Console()
    console.print("\n[bold cyan]🚀 Starting Comprehensive Multi-Agent Topology Experiments (Parallel Mode)[/bold cyan]")
    
    import random
    random.seed(42)
    
    num_tasks = 50
    console.print(f"Loading and sampling {num_tasks} tasks with varied difficulty from MBPP Sanitzed dataset...")
    full_dataset = load_dataset("mbpp", "sanitized", split="test")
    
    # Proxy difficulty by prompt length
    sorted_dataset = sorted(list(full_dataset), key=lambda x: len(x.get('prompt', '')))
    
    if len(sorted_dataset) >= num_tasks:
        hard = int(num_tasks * 0.6)
        med = int(num_tasks * 0.2)
        easy = num_tasks - hard - med
        # Biased sample: 60% hardest, 20% median, 20% easiest
        dataset = sorted_dataset[-hard:] + sorted_dataset[len(sorted_dataset)//2 - med//2 : len(sorted_dataset)//2 - med//2 + med] + sorted_dataset[:easy]
        random.shuffle(dataset)
    else:
        dataset = sorted_dataset
    
    graphs = {
        "G0: Baseline (0-shot)": {"builder": build_g0_baseline, "roles": ["universal_agent"]},
        "G1: Waterfall (Linear)": {"builder": build_g1_waterfall, "roles": ["planner", "coder", "reviewer", "tester"]},
        "G2: AgentCoder (Loop)": {"builder": build_g2_agentcoder, "roles": ["planner", "coder", "tester"]},
        "G3: MapCoder (Cycle)": {"builder": build_g3_mapcoder, "roles": ["planner", "coder", "reviewer", "tester"]},
        "G4: Parallel Judge": {"builder": build_g4_parallel_judge, "roles": ["planner", "coder", "judge", "tester"]},
        "G5: Adversarial Debate": {"builder": build_g5_adversarial_debate, "roles": ["planner", "coder", "red_team", "blue_team", "tester"]},
        "G6: Hierarchical Setup": {"builder": build_g6_hierarchical, "roles": ["manager", "worker", "aggregator", "tester"]},
        "G7: Mental Simulator": {"builder": build_g7_mental_simulator, "roles": ["planner", "coder", "simulator", "tester"]},
    }
    
    # Store aggregated results
    summary_results = {}
    
    # Determine safe worker count for 16GB M4
    MAX_WORKERS = 10  # Increased for API concurrency
    
    # Load previous progress to skip tasks
    existing_results = load_existing_results()
    if existing_results:
        console.print(f"🔄 Found existing progress. Resume mode enabled.")
    
    for graph_name, graph_info in graphs.items():
        console.print(f"\n[bold yellow]Evaluating Topology: {graph_name}[/bold yellow]")
        
        roles = graph_info["roles"]
        builder_func = graph_info["builder"]
        base_profile = {r: 1 for r in roles}
        
        graph_results = []
        tasks_to_run = []
        
        # Check against existing results to skip duplicates
        for row in dataset:
            task_id = f"MBPP-{row['task_id']}"
            if graph_name in existing_results and task_id in existing_results[graph_name]:
                graph_results.append(existing_results[graph_name][task_id])
            else:
                tasks_to_run.append(row)
        
        if len(graph_results) > 0:
            console.print(f"  ⏭️ Skipping {len(graph_results)} tasks already completed for {graph_name}")

        # Parallelize remaining task evaluation
        if tasks_to_run:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_task = {
                    executor.submit(evaluate_task, row, builder_func, graph_name, roles, base_profile, console): row 
                    for row in tasks_to_run
                }
                
                # Use tqdm for progress tracking
                for future in tqdm(as_completed(future_to_task), total=len(tasks_to_run), desc=f"Remaining tasks for {graph_name}"):
                    result = future.result()
                    if result:
                        graph_results.append(result)
        
        if len(graph_results) > 0:
            pass_rate = sum(1 for r in graph_results if r["passed"]) / len(graph_results)
            avg_tokens = sum(r["tokens"] for r in graph_results) / len(graph_results)
            
            # Count the percentage of tasks where ALL agents were strictly incentivized
            fully_stable_tasks = sum(1 for r in graph_results if all(r["stability"].values()))
            stability_rate = fully_stable_tasks / len(graph_results)
            
            summary_results[graph_name] = {
                "Pass Rate": f"{pass_rate*100:.1f}%",
                "Avg Tokens": f"{avg_tokens:.0f}",
                "Nash Stability": f"{stability_rate*100:.1f}%"
            }

    # Export to CSV final results
    csv_file = Path("data/results.csv")
    csv_file.parent.mkdir(exist_ok=True)
    with open(csv_file, "w") as f:
        f.write("Topology,Pass_Rate,Avg_Tokens,Nash_Stability\n")
        for graph_name, metrics in summary_results.items():
            f.write(f'"{graph_name}","{metrics["Pass Rate"]}",{metrics["Avg Tokens"]},"{metrics["Nash Stability"]}"\n')
            
    # Print the cool table
    console.print("\n[bold green]==== FINAL COMPREHENSIVE EXPERIMENT RESULTS ====[/bold green]")
    table = Table(title="Topology Performance Matrix (MBPP)", style="cyan")
    table.add_column("Topology Configuration", justify="left", style="white", no_wrap=True)
    table.add_column("Pass@1 Accuracy", justify="center", style="green")
    table.add_column("Average Token Cost", justify="center", style="yellow")
    table.add_column("Nash Incentive Stability", justify="center", style="magenta")
    
    for graph_name, metrics in summary_results.items():
        table.add_row(
            graph_name, 
            metrics["Pass Rate"], 
            metrics["Avg Tokens"], 
            metrics["Nash Stability"]
        )
        
    console.print(table)
    
if __name__ == "__main__":
    run_all_experiments()
