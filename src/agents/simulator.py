from langchain_core.prompts import ChatPromptTemplate
from src.core.model import get_llm, get_system_prompt_modifier
from src.core.state import TaskState

class MentalSimulatorNode:
    def __init__(self, effort_level: int = 1):
        self.llm = get_llm(effort_level)
        self.modifier = get_system_prompt_modifier(effort_level)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are the Mental Simulator. Read this python code and mentally execute it step-by-step for the given unit tests. If it passes, simply reply 'PASS'. If you spot a logical error during dry-run, output the error details to fix." + self.modifier),
            ("user", "Task:\n{task_description}\n\nCode to simulate:\n{code}\n\nTests:\n{tests}")
        ])
        
    def __call__(self, state: TaskState):
        chain = self.prompt | self.llm
        result = chain.invoke({
            "task_description": state.get("task_description", ""),
            "tests": "\n".join(state.get("tests", [])),
            "code": state.get("code", "")
        })
        
        sim_feedback = result.content.strip()
        
        return {
            "token_usage": {"simulator": result.response_metadata.get('token_usage', {}).get('total_tokens', 0)},
            "simulation_feedback": sim_feedback
        }
