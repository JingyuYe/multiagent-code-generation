import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graphs.g0_baseline import build_g0_baseline
from src.graphs.g1_waterfall import build_g1_waterfall
from src.graphs.g2_agentcoder import build_g2_agentcoder
from src.graphs.g2_5_reviewer_repair import build_g2_reviewer_repair
from src.graphs.g3_mapcoder import build_g3_mapcoder
from src.graphs.g4_parallel_judge import build_g4_parallel_judge
from src.graphs.g5_adversarial_debate import build_g5_adversarial_debate
from src.graphs.g6_hierarchical import build_g6_hierarchical
from src.graphs.g7_mental_simulator import build_g7_mental_simulator

graphs = {
    1: build_g0_baseline(),
    2: build_g1_waterfall(),
    3: build_g2_agentcoder(),
    4: build_g2_reviewer_repair(),
    5: build_g3_mapcoder(),
    6: build_g4_parallel_judge(),
    7: build_g5_adversarial_debate(),
    8: build_g6_hierarchical()
}

for i, graph in graphs.items():
    try:
        png_data = graph.get_graph().draw_mermaid_png()
        with open(f"graph_structure-{i}.png", "wb") as f:
            f.write(png_data)
        print(f"Generated graph_structure-{i}.png")
    except Exception as e:
        print(f"Failed to generate {i}: {e}")
