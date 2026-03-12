from langchain_core.prompts import ChatPromptTemplate
from src.core.model import get_llm, get_system_prompt_modifier
from src.core.state import TaskState

class ReviewerNode:
    def __init__(self, effort_level: int = 1):
        self.llm = get_llm(effort_level)
        self.modifier = get_system_prompt_modifier(effort_level)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are the Reviewer. Statically analyze the generated Python code for logical errors, edge cases, and style prior to execution." + self.modifier),
            ("user", "Task:\n{task_description}\n\nCode to review:\n{code}\n\nProvide review comments.")
        ])
        
    def __call__(self, state: TaskState):
        chain = self.prompt | self.llm
        
        task_desc = state.get("task_description", "")
        code = state.get("code", "No code provided.")
        
        result = chain.invoke({
            "task_description": task_desc,
            "code": code
        })
        
        estimated_tokens = len(result.content.split())
        
        reviews = state.get("review_comments", [])
        reviews.append(result.content)
        
        return {
            "review_comments": reviews,
            "token_usage": {"reviewer": estimated_tokens}
        }
