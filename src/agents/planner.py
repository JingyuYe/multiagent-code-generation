from langchain_core.prompts import ChatPromptTemplate
from src.core.model import get_llm, get_system_prompt_modifier
from src.core.state import TaskState

class PlannerNode:
    def __init__(self, effort_level: int = 1):
        self.llm = get_llm(effort_level)
        self.modifier = get_system_prompt_modifier(effort_level)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are the Planner. Break down the task into a step-by-step logic plan." + self.modifier),
            ("user", "Task:\n{task_description}")
        ])
        
    def __call__(self, state: TaskState):
        chain = self.prompt | self.llm
        result = chain.invoke({"task_description": state.get("task_description", "")})
        
        # In a real implementation we would capture actual tokens used 
        # (e.g. using Langchain's get_openai_callback, though Ollama might need custom tracking)
        estimated_tokens = len(result.content.split())
        
        return {
            "plan": result.content,
            "token_usage": {"planner": estimated_tokens}
        }
