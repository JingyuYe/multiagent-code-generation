from langgraph.graph import StateGraph
from src.core.state import TaskState
from src.agents.planner import PlannerNode
from src.agents.judge import ParallelCoderNode, JudgeNode
from src.agents.tester import TesterNode

def build_g4_parallel_judge(effort_profile: dict = None):
    if effort_profile is None:
        effort_profile = {"planner": 1, "coder": 1, "judge": 1, "tester": 1}

    builder = StateGraph(TaskState)
    
    planner = PlannerNode(effort_profile.get("planner", 1))
    coder_a = ParallelCoderNode(effort_profile.get("coder", 1))
    coder_b = ParallelCoderNode(effort_profile.get("coder", 1))
    coder_c = ParallelCoderNode(effort_profile.get("coder", 1))
    judge = JudgeNode(effort_profile.get("judge", 1))
    tester = TesterNode(effort_profile.get("tester", 1))
    
    builder.add_node("Planner", planner)
    builder.add_node("CoderA", coder_a)
    builder.add_node("CoderB", coder_b)
    builder.add_node("CoderC", coder_c)
    builder.add_node("Judge", judge)
    builder.add_node("Tester", tester)
    
    builder.set_entry_point("Planner")
    
    # Fan out from Planner to 3 parallel coders
    builder.add_edge("Planner", "CoderA")
    builder.add_edge("Planner", "CoderB")
    builder.add_edge("Planner", "CoderC")
    
    # Fan in to Judge
    builder.add_edge("CoderA", "Judge")
    builder.add_edge("CoderB", "Judge")
    builder.add_edge("CoderC", "Judge")
    
    builder.add_edge("Judge", "Tester")
    builder.set_finish_point("Tester")
    
    return builder.compile()
