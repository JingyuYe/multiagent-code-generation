from langgraph.graph import StateGraph
from src.core.state import TaskState
from src.agents.hierarchy import ManagerNode, WorkerNode, AggregatorNode
from src.agents.tester import TesterNode

def build_g6_hierarchical(effort_profile: dict = None):
    if effort_profile is None:
        effort_profile = {"manager": 1, "worker": 1, "aggregator": 1, "tester": 1}

    builder = StateGraph(TaskState)
    
    manager = ManagerNode(effort_profile.get("manager", 1))
    worker = WorkerNode(effort_profile.get("worker", 1))
    aggregator = AggregatorNode(effort_profile.get("aggregator", 1))
    tester = TesterNode(effort_profile.get("tester", 1))
    
    builder.add_node("Manager", manager)
    builder.add_node("Worker", worker)
    builder.add_node("Aggregator", aggregator)
    builder.add_node("Tester", tester)
    
    builder.set_entry_point("Manager")
    builder.add_edge("Manager", "Worker")
    builder.add_edge("Worker", "Aggregator")
    builder.add_edge("Aggregator", "Tester")
    builder.set_finish_point("Tester")
    
    return builder.compile()
