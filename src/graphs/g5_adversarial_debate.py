from langgraph.graph import StateGraph
from src.core.state import TaskState
from src.agents.planner import PlannerNode
from src.agents.coder import CoderNode
from src.agents.debater import RedTeamDebaterNode, BlueTeamDebaterNode
from src.agents.tester import TesterNode

def debate_router(state: TaskState):
    if state.get("iterations", 0) >= 2: # Max 2 rounds of debate
        return "Tester"
    return "RedTeam"

def build_g5_adversarial_debate(effort_profile: dict = None):
    if effort_profile is None:
        effort_profile = {"planner": 1, "coder": 1, "red_team": 1, "blue_team": 1, "tester": 1}

    builder = StateGraph(TaskState)
    
    planner = PlannerNode(effort_profile.get("planner", 1))
    coder = CoderNode(effort_profile.get("coder", 1))
    red_team = RedTeamDebaterNode(effort_profile.get("red_team", 1))
    blue_team = BlueTeamDebaterNode(effort_profile.get("blue_team", 1))
    tester = TesterNode(effort_profile.get("tester", 1))
    
    builder.add_node("Planner", planner)
    builder.add_node("Coder", coder)
    builder.add_node("RedTeam", red_team)
    builder.add_node("BlueTeam", blue_team)
    builder.add_node("Tester", tester)
    
    builder.set_entry_point("Planner")
    builder.add_edge("Planner", "Coder")
    builder.add_edge("Coder", "RedTeam")
    builder.add_edge("RedTeam", "BlueTeam")
    
    builder.add_conditional_edges(
        "BlueTeam",
        debate_router,
        {
            "Tester": "Tester",
            "RedTeam": "RedTeam"
        }
    )
    
    builder.set_finish_point("Tester")
    
    return builder.compile()
