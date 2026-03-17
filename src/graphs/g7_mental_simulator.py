from langgraph.graph import StateGraph
from src.core.state import TaskState
from src.agents.planner import PlannerNode
from src.agents.coder import CoderNode
from src.agents.simulator import MentalSimulatorNode
from src.agents.tester import TesterNode

def routing_decision(state: TaskState):
    feedback = state.get("simulation_feedback", "")
    if "PASS" in feedback.upper() and len(feedback) < 20: 
        return "Tester" # The simulator thinks it's completely fine, send to actual runtime test
    return "Coder" # The simulator found an issue, send back to coder

def build_g7_mental_simulator(effort_profile: dict = None):
    if effort_profile is None:
        effort_profile = {"planner": 1, "coder": 1, "simulator": 1, "tester": 1}

    builder = StateGraph(TaskState)
    
    planner = PlannerNode(effort_profile.get("planner", 1))
    coder = CoderNode(effort_profile.get("coder", 1))
    simulator = MentalSimulatorNode(effort_profile.get("simulator", 1))
    tester = TesterNode(effort_profile.get("tester", 1))
    
    builder.add_node("Planner", planner)
    builder.add_node("Coder", coder)
    builder.add_node("Simulator", simulator)
    builder.add_node("Tester", tester)
    
    builder.set_entry_point("Planner")
    builder.add_edge("Planner", "Coder")
    builder.add_edge("Coder", "Simulator")
    
    builder.add_conditional_edges(
        "Simulator",
        routing_decision,
        {
            "Tester": "Tester",
            "Coder": "Coder"
        }
    )
    
    builder.add_edge("Tester", "Planner") # Normal mapcoder loop on real failure
    
    return builder.compile()
