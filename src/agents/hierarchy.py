import json
from langchain_core.prompts import ChatPromptTemplate
from src.core.model import get_llm, get_system_prompt_modifier
from src.core.state import TaskState

class ManagerNode:
    def __init__(self, effort_level: int = 1):
        self.llm = get_llm(effort_level)
        self.modifier = get_system_prompt_modifier(effort_level)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are the Logic Manager. Break the task down into exact sub-task function names that need to be written (e.g. ['validate_input', 'compute_matrix', 'format_output']). Output ONLY a JSON list of strings." + self.modifier),
            ("user", "Task:\n{task_description}\nTests:\n{tests}\n\nList the subtasks as JSON.")
        ])
        
    def __call__(self, state: TaskState):
        chain = self.prompt | self.llm
        result = chain.invoke({
            "task_description": state.get("task_description", ""),
            "tests": "\n".join(state.get("tests", []))
        })
        
        try:
            subtasks = json.loads(result.content)
            if not isinstance(subtasks, list): subtasks = ["main_function"]
        except Exception:
            subtasks = ["main_function"]
            
        return {
            "token_usage": {"manager": result.response_metadata.get('token_usage', {}).get('total_tokens', 0)},
            "subtasks": subtasks
        }

class WorkerNode:
    def __init__(self, effort_level: int = 1):
        self.llm = get_llm(effort_level)
        self.modifier = get_system_prompt_modifier(effort_level)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a Worker. Write ONLY the Python code for your assigned subtask. Do not write the full program, just your function." + self.modifier),
            ("user", "Task:\n{task_description}\n\nYour Subtask is: {focus}\n\nGenerate exactly your python function.")
        ])
        
    def __call__(self, state: TaskState):
        subtasks = state.get("subtasks", [])
        codes = []
        tokens = 0
        
        chain = self.prompt | self.llm
        
        # In a real parallel map-reduce, this would be parallel node expansions.
        # For simplicity in testing, we map across them here iteratively.
        for task in subtasks:
            result = chain.invoke({
                "task_description": state.get("task_description", ""),
                "focus": task
            })
            tokens += result.response_metadata.get('token_usage', {}).get('total_tokens', 0)
            
            code = result.content
            if code.startswith("```python"): code = code[9:]
            if code.startswith("```"): code = code[3:]
            if code.endswith("```"): code = code[:-3]
            codes.append(code.strip())
            
        return {
            "token_usage": {"worker": tokens},
            "subtask_code": codes
        }

class AggregatorNode:
    def __init__(self, effort_level: int = 1):
        self.llm = get_llm(effort_level)
        self.modifier = get_system_prompt_modifier(effort_level)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an Aggregator. Take these fragmented sub-functions and weave them together into a final robust script that passes the tests. Output ONLY the unified python code." + self.modifier),
            ("user", "Task:\n{task_description}\nTests:\n{tests}\n\nSub-functions:\n{snippets}\n\nAssemble the final code.")
        ])
        
    def __call__(self, state: TaskState):
        snippets = "\n\n# --- SNIPPET ---\n\n".join(state.get("subtask_code", []))
        
        chain = self.prompt | self.llm
        result = chain.invoke({
            "task_description": state.get("task_description", ""),
            "tests": "\n".join(state.get("tests", [])),
            "snippets": snippets
        })
        
        code = result.content
        if code.startswith("```python"): code = code[9:]
        if code.startswith("```"): code = code[3:]
        if code.endswith("```"): code = code[:-3]
        
        return {
            "token_usage": {"aggregator": result.response_metadata.get('token_usage', {}).get('total_tokens', 0)},
            "code": code.strip()
        }
