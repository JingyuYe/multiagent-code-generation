from langchain_core.prompts import ChatPromptTemplate
from src.core.model import get_llm, get_system_prompt_modifier
from src.core.state import TaskState

class CoderNode:
    def __init__(self, effort_level: int = 1):
        self.llm = get_llm(effort_level)
        self.modifier = get_system_prompt_modifier(effort_level)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are the Coder. Translate the logic plan directly into Python code. Pay close attention to the provided Unit Tests and ensure your function name and arguments exactly match what the tests expect." + self.modifier),
            ("user", "Task:\n{task_description}\n\nUnit Tests (ensure your function name matches these):\n{tests}\n\nPlan:\n{plan}\n\nExisting Code:\n{existing_code}\n\nTest Results (if any):\n{test_results}\n\nReviewer Comments (if any):\n{review_comments}\n\nGenerate strictly the Python code.")
        ])
        
    def __call__(self, state: TaskState):
        chain = self.prompt | self.llm
        
        # Format the context arguments
        task_desc = state.get("task_description", "")
        plan = state.get("plan", "No plan provided.")
        existing_code = state.get("code", "")
        tests = "\n".join(state.get("tests", []))
        test_results = "\n".join(state.get("test_results", []))
        review_comments = "\n".join(state.get("review_comments", []))
        
        result = chain.invoke({
            "task_description": task_desc,
            "tests": tests,
            "plan": plan, 
            "existing_code": existing_code,
            "test_results": test_results,
            "review_comments": review_comments
        })
        
        # Estimate usage for mock purposes
        estimated_tokens = len(result.content.split())
        code = result.content
        if "```python" in code:
            code = code.split("```python")[-1].split("```")[0]
        elif "```" in code:
            code = code.split("```")[-1].split("```")[0]
        code = code.strip()
        
        return {
            "code": code,
            "token_usage": {"coder": estimated_tokens}
        }
