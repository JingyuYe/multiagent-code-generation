import json
import pandas as pd
from collections import defaultdict

raw_results = []
try:
    with open('data/raw_results.jsonl', 'r') as f:
        for line in f:
            if line.strip():
                try:
                    raw_results.append(json.loads(line))
                except:
                    pass
except FileNotFoundError:
    pass

if raw_results:
    summary_results = {
        "G0: Baseline (0-shot)": {"Pass_Rate": "0.0%", "Avg_Tokens": 0, "Nash_Stability": "0.0%"},
        "G1: Waterfall (Linear)": {"Pass_Rate": "0.0%", "Avg_Tokens": 0, "Nash_Stability": "0.0%"},
        "G2: AgentCoder (Loop)": {"Pass_Rate": "0.0%", "Avg_Tokens": 0, "Nash_Stability": "0.0%"},
        "G3: MapCoder (Cycle)": {"Pass_Rate": "0.0%", "Avg_Tokens": 0, "Nash_Stability": "0.0%"},
        "G4: Parallel Judge": {"Pass_Rate": "0.0%", "Avg_Tokens": 0, "Nash_Stability": "0.0%"},
        "G5: Adversarial Debate": {"Pass_Rate": "0.0%", "Avg_Tokens": 0, "Nash_Stability": "0.0%"},
        "G6: Hierarchical Setup": {"Pass_Rate": "0.0%", "Avg_Tokens": 0, "Nash_Stability": "0.0%"}
    }
    
    # Group by topology
    topology_groups = defaultdict(list)
    for r in raw_results:
        topology_groups[r['topology']].append(r)
        
    for graph_name, graph_results in topology_groups.items():
        if len(graph_results) > 0:
            pass_rate = sum(1 for r in graph_results if r.get("passed", False)) / len(graph_results)
            avg_tokens = sum(r.get("tokens", 0) for r in graph_results) / len(graph_results)
            
            # Count fully stable tasks
            if 'universal_agent' in graph_results[0].get('stability', {}):
                stability_rate = 1.0 # Baseline is always stable
            else:
                fully_stable_tasks = sum(1 for r in graph_results if all(r.get("stability", {}).values()))
                stability_rate = fully_stable_tasks / len(graph_results)
            
            summary_results[graph_name] = {
                "Pass_Rate": f"{pass_rate*100:.1f}%",
                "Avg_Tokens": int(avg_tokens),
                "Nash_Stability": f"{stability_rate*100:.1f}%"
            }
    
    # Create DataFrame and export
    df = pd.DataFrame([{
        "Topology": k,
        "Pass_Rate": v["Pass_Rate"],
        "Avg_Tokens": v["Avg_Tokens"],
        "Nash_Stability": v["Nash_Stability"]
    } for k, v in summary_results.items()])
    
    df.to_csv('data/results.csv', index=False)
    print(f"Generated data/results.csv from {len(raw_results)} partial raw results.")
else:
    print("No raw results found.")
