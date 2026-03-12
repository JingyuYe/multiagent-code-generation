from langchain_core.prompts import ChatPromptTemplate
from src.core.model import get_llm, get_system_prompt_modifier
from src.core.state import TaskState
from src.execution.sandbox import run_tests_in_sandbox

class TesterNode:
    def __init__(self, effort_level: int = 1):
        self.llm = get_llm(effort_level)
        self.modifier = get_system_prompt_modifier(effort_level)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are the Tester. Look at the execution trace and isolate any runtime errors. Identify what went wrong with the logic." + self.modifier),
            ("user", "Task:\n{task_description}\n\nCode:\n{code}\n\nExecution Trace:\n{trace}\n\nAnalyze the outcome.")
        ])
        
    def __call__(self, state: TaskState):
        chain = self.prompt | self.llm
        
        task_desc = state.get("task_description", "")
        code = state.get("code", "")
        tests = state.get("tests", [])
        
        # 1. Box execution
        passed, trace = run_tests_in_sandbox(code, tests)
        
        # 2. LLM Analysis
        result = chain.invoke({
            "task_description": task_desc,
            "code": code,
            "trace": trace
        })
        
        estimated_tokens = len(result.content.split())
        
        test_results = state.get("test_results", [])
        test_results.append(f"Passed: {passed}\nTrace:\n{trace}\nTester Analysis: {result.content}")
        
        return {
            "test_results": test_results,
            "test_passed": passed,
            "iterations": 1,
            "token_usage": {"tester": estimated_tokens}
        }
