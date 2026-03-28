from langgraph.graph import StateGraph, END
from src.core.state import TaskState
from src.agents.planner import PlannerNode
from src.agents.coder import CoderNode
from src.agents.reviewer import ReviewerNode
from src.agents.tester import TesterNode

MAX_ITERATIONS = 3

def route_after_tester(state: TaskState):
    """
    If tests passed, stop.
    If max iterations reached, stop.
    Otherwise send to Reviewer for failure diagnosis.
    """
    if state.get("test_passed", False):
        return END

    if state.get("iterations", 0) >= MAX_ITERATIONS:
        return END

    return "Reviewer"

def build_g2_reviewer_repair(effort_profile: dict = None):
    """
    G2+Reviewer:
    Planner -> Coder -> Tester
                     if fail
                        ↓
                     Reviewer -> Coder -> Tester
    """
    if effort_profile is None:
        effort_profile = {
            "planner": 1,
            "coder": 1,
            "reviewer": 1,
            "tester": 1,
        }

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
    builder.add_edge("Coder", "Tester")

    builder.add_conditional_edges("Tester", route_after_tester, path_map={"Reviewer": "Reviewer", END: END})

    builder.add_edge("Reviewer", "Coder")

    return builder.compile()