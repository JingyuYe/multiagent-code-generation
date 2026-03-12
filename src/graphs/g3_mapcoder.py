from langgraph.graph import StateGraph, END
from src.core.state import TaskState
from src.agents.planner import PlannerNode
from src.agents.coder import CoderNode
from src.agents.reviewer import ReviewerNode
from src.agents.tester import TesterNode

MAX_ITERATIONS = 3

def should_continue(state: TaskState):
    if state.get("test_passed", False):
        return END
    
    if state.get("iterations", 0) >= MAX_ITERATIONS:
        return END
        
    return "Planner"

def build_g3_mapcoder(effort_profile: dict = None):
    """
    G3: MapCoder-Style Cycle. Retrieval-planning-coding-debugging under a fixed budget.
    We simulate this by cycling back to the Planner upon failure: P -> C -> R -> T -> P...
    """
    if effort_profile is None:
        effort_profile = {"planner": 1, "coder": 1, "reviewer": 1, "tester": 1}

    builder = StateGraph(TaskState)
    
    planner = PlannerNode(effort_profile.get("planner", 1))
    coder = CoderNode(effort_profile.get("coder", 1))
    reviewer = ReviewerNode(effort_profile.get("reviewer", 1))
    tester = TesterNode(effort_profile.get("tester", 1))
    
    builder.add_node("Planner", planner)
    builder.add_node("Coder", coder)
    builder.add_node("Reviewer", reviewer)
    builder.add_node("Tester", tester)
    
    builder.set_entry_point("Planner")
    builder.add_edge("Planner", "Coder")
    builder.add_edge("Coder", "Reviewer")
    builder.add_edge("Reviewer", "Tester")
    
    # Conditional edge to loop back to Planner if the code fails to pass tests
    builder.add_conditional_edges("Tester", should_continue)
    
    return builder.compile()
