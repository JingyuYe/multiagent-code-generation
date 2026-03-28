import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langchain_core.runnables.graph import MermaidDrawMethod

from src.graphs.g2_5_reviewer_repair import build_g2_reviewer_repair

graph = build_g2_reviewer_repair()

try:
    png_data = graph.get_graph().draw_mermaid_png()
    with open("graph_structure-4.png", "wb") as f:
        f.write(png_data)
    print("Generated graph_structure-4.png")
except Exception as e:
    try:
        print("Retrying with Pyppeteer...")
        png_data = graph.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.PYPPETEER)
        with open("graph_structure-4.png", "wb") as f:
            f.write(png_data)
        print("Generated graph_structure-4.png")
    except Exception as e2:
        print(f"Failed again: {e2}")
