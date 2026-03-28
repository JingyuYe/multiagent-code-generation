from langchain_core.prompts import ChatPromptTemplate
from src.core.model import get_llm, get_system_prompt_modifier
from src.core.state import TaskState

class PlannerNode:
    def __init__(self, effort_level: int = 1):
        self.llm = get_llm(effort_level)
        self.modifier = get_system_prompt_modifier(effort_level)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are the Planner. Break down the task into a step-by-step logic plan. If existing code and test results are provided from a previous failed attempt, analyze them to suggest a better approach." + self.modifier),
            ("user", "Task:\n{task_description}\n\nExisting Code (if retry):\n{existing_code}\n\nTest Results (if retry):\n{test_results}\n\nReviewer Comments:\n{review_comments}\n\nProvide the new plan:")
        ])
        
    def __call__(self, state: TaskState):
        chain = self.prompt | self.llm
        
        task_desc = state.get("task_description", "")
        existing_code = state.get("code", "")
        test_results = "\n".join(state.get("test_results", []))
        review_comments = "\n".join(state.get("review_comments", []))
        
        result = chain.invoke({
            "task_description": task_desc,
            "existing_code": existing_code,
            "test_results": test_results,
            "review_comments": review_comments
        })
        
        # In a real implementation we would capture actual tokens used 
        # (e.g. using Langchain's get_openai_callback, though Ollama might need custom tracking)
        estimated_tokens = len(result.content.split())
        
        return {
            "plan": result.content,
            "token_usage": {"planner": estimated_tokens}
        }
