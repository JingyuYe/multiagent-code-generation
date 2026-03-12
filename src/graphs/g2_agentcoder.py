from langgraph.graph import StateGraph, END
from src.core.state import TaskState
from src.agents.planner import PlannerNode
from src.agents.coder import CoderNode
from src.agents.tester import TesterNode

MAX_ITERATIONS = 3

def should_continue(state: TaskState):
    if state.get("test_passed", False):
        return END
    
    if state.get("iterations", 0) >= MAX_ITERATIONS:
        return END
        
    return "Coder"

def build_g2_agentcoder(effort_level: int = 1):
    """
    G2: AgentCoder-Style Loop. Iterative code-test-revise until tests pass or budget exhausted.
    Task -> PL -> C <---> T
    """
    builder = StateGraph(TaskState)
    
    planner = PlannerNode(effort_level)
    coder = CoderNode(effort_level)
    tester = TesterNode(effort_level)
    
    builder.add_node("Planner", planner)
    builder.add_node("Coder", coder)
    builder.add_node("Tester", tester)
    
    builder.set_entry_point("Planner")
    builder.add_edge("Planner", "Coder")
    builder.add_edge("Coder", "Tester")
    
    # Conditional edge to loop back to Coder on failure
    builder.add_conditional_edges("Tester", should_continue)
    
    return builder.compile()
