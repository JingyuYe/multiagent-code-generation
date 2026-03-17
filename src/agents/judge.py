from langchain_core.prompts import ChatPromptTemplate
from src.core.model import get_llm, get_system_prompt_modifier
from src.core.state import TaskState

class ParallelCoderNode:
    """
    A variant of the Coder that appends its output to code_candidates instead of replacing code.
    """
    def __init__(self, effort_level: int = 1):
        self.llm = get_llm(effort_level)
        self.modifier = get_system_prompt_modifier(effort_level)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a Coder. Translate the logic plan directly into Python code. Pay close attention to Unit Tests." + self.modifier),
            ("user", "Task:\n{task_description}\nPlan:\n{plan}\nTests:\n{tests}\n\nGenerate strictly Python code without formatting tags.")
        ])
        
    def __call__(self, state: TaskState):
        chain = self.prompt | self.llm
        result = chain.invoke({
            "task_description": state.get("task_description", ""),
            "tests": "\n".join(state.get("tests", [])),
            "plan": state.get("plan", "No plan provided.")
        })
        
        # Strip code bounds
        code = result.content
        if code.startswith("```python"): code = code[9:]
        if code.startswith("```"): code = code[3:]
        if code.endswith("```"): code = code[:-3]
        
        return {
            "token_usage": {"coder": result.response_metadata.get('token_usage', {}).get('total_tokens', 0)},
            "code_candidates": [code.strip()]
        }

class JudgeNode:
    """
    Evaluates multiple code candidates and selects the best one.
    """
    def __init__(self, effort_level: int = 1):
        self.llm = get_llm(effort_level)
        self.modifier = get_system_prompt_modifier(effort_level)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a Senior Engineer Judge. You are given multiple Python implementations. Choose the best, most robust one that passes the tests. Output ONLY the winning python code." + self.modifier),
            ("user", "Task:\n{task_description}\n\nTests:\n{tests}\n\nCandidates:\n{candidates}\n\nSelect the best code.")
        ])
        
    def __call__(self, state: TaskState):
        candidates = state.get("code_candidates", [])
        candidates_text = "\n\n=== CANDIDATE ===\n".join(candidates)
        
        chain = self.prompt | self.llm
        result = chain.invoke({
            "task_description": state.get("task_description", ""),
            "tests": "\n".join(state.get("tests", [])),
            "candidates": candidates_text
        })
        
        code = result.content
        if code.startswith("```python"): code = code[9:]
        if code.startswith("```"): code = code[3:]
        if code.endswith("```"): code = code[:-3]
        
        return {
            "token_usage": {"judge": result.response_metadata.get('token_usage', {}).get('total_tokens', 0)},
            "code": code.strip()
        }
