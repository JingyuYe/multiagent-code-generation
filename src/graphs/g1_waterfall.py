from langgraph.graph import StateGraph, END
from src.core.state import TaskState
from src.agents.planner import PlannerNode
from src.agents.coder import CoderNode
from src.agents.reviewer import ReviewerNode
from src.agents.tester import TesterNode

def build_g1_waterfall(effort_level: int = 1):
    """
    G1: Waterfall Pipeline. Sequential PL -> C -> R -> T with minimal feedback.
    """
    builder = StateGraph(TaskState)
    
    # Instantiate agents at specified effort level
    planner = PlannerNode(effort_level)
    coder = CoderNode(effort_level)
    reviewer = ReviewerNode(effort_level)
    tester = TesterNode(effort_level)
    
    # Add nodes
    builder.add_node("Planner", planner)
    builder.add_node("Coder", coder)
    builder.add_node("Reviewer", reviewer)
    builder.add_node("Tester", tester)
    
    # Define linear edges
    builder.set_entry_point("Planner")
    builder.add_edge("Planner", "Coder")
    builder.add_edge("Coder", "Reviewer")
    builder.add_edge("Reviewer", "Tester")
    builder.add_edge("Tester", END)
    
    return builder.compile()
