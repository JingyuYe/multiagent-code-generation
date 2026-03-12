from langgraph.graph import StateGraph, END
from src.core.state import TaskState
from langchain_core.prompts import ChatPromptTemplate
from src.core.model import get_llm, get_system_prompt_modifier
from src.execution.sandbox import run_tests_in_sandbox

class UniversalAgentNode:
    def __init__(self, effort_level: int = 1):
        self.llm = get_llm(effort_level)
        self.modifier = get_system_prompt_modifier(effort_level)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert Python coder. Given a task, write solely the Python code to solve it. Provide ONLY the code in a standard markdown python block. No other explanation." + self.modifier),
            ("user", "Task:\n{task_description}\n\nExisting Code:\n{existing_code}\n\nGenerate strictly the Python code in a markdown block." )
        ])
        
    def __call__(self, state: TaskState):
        chain = self.prompt | self.llm
        
        task_desc = state.get("task_description", "")
        existing_code = state.get("code", "")
        
        result = chain.invoke({
            "task_description": task_desc,
            "existing_code": existing_code
        })
        
        estimated_tokens = len(result.content.split())
        
        code = result.content
        if "```python" in code:
            code = code.split("```python")[-1].split("```")[0]
        elif "```" in code:
            code = code.split("```")[-1].split("```")[0]
        code = code.strip()
        
        tests = state.get("tests", [])
        passed, trace = run_tests_in_sandbox(code, tests)
        
        test_results = state.get("test_results", [])
        test_results.append(f"Passed: {passed}\nTrace:\n{trace}")
        
        return {
            "code": code,
            "test_results": test_results,
            "test_passed": passed,
            "iterations": 1,
            "token_usage": {"universal_agent": estimated_tokens}
        }

def build_g0_baseline(effort_level: int = 1):
    """
    G0: Single-Agent Baseline. One-shot solution.
    """
    builder = StateGraph(TaskState)
    
    agent = UniversalAgentNode(effort_level)
    
    builder.add_node("Agent", agent)
    
    builder.set_entry_point("Agent")
    builder.add_edge("Agent", END)
    
    return builder.compile()
