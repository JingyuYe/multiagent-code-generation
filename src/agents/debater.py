from langchain_core.prompts import ChatPromptTemplate
from src.core.model import get_llm, get_system_prompt_modifier
from src.core.state import TaskState

class RedTeamDebaterNode:
    def __init__(self, effort_level: int = 1):
        self.llm = get_llm(effort_level)
        self.modifier = get_system_prompt_modifier(effort_level)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are the Red Team (Adversary). Your ONLY goal is to find edge cases, logic flaws, or security issues in this Python code. Be critical but precise. Output only your critique." + self.modifier),
            ("user", "Task:\n{task_description}\nTests:\n{tests}\nCode:\n{code}\nDebate History:\n{history}\n\nCritique the code.")
        ])
        
    def __call__(self, state: TaskState):
        chain = self.prompt | self.llm
        result = chain.invoke({
            "task_description": state.get("task_description", ""),
            "tests": "\n".join(state.get("tests", [])),
            "code": state.get("code", ""),
            "history": "\n".join(state.get("debate_history", []))
        })
        return {
            "token_usage": {"red_team": result.response_metadata.get('token_usage', {}).get('total_tokens', 0)},
            "debate_history": ["Red Team: " + result.content]
        }

class BlueTeamDebaterNode:
    def __init__(self, effort_level: int = 1):
        self.llm = get_llm(effort_level)
        self.modifier = get_system_prompt_modifier(effort_level)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are the Blue Team (Defender/Coder). Act upon the Red Team's critique. If they found a flaw, fix the code and return only the NEW python code. If the code is fine, return the OLD code." + self.modifier),
            ("user", "Task:\n{task_description}\nTests:\n{tests}\nOld Code:\n{code}\nCritique:\n{critique}\n\nReturn ONLY the exact python code responding to the critique.")
        ])
        
    def __call__(self, state: TaskState):
        history = state.get("debate_history", [])
        critique = history[-1] if history else "No critique."
        
        chain = self.prompt | self.llm
        result = chain.invoke({
            "task_description": state.get("task_description", ""),
            "tests": "\n".join(state.get("tests", [])),
            "code": state.get("code", ""),
            "critique": critique
        })
        
        code = result.content
        if code.startswith("```python"): code = code[9:]
        if code.startswith("```"): code = code[3:]
        if code.endswith("```"): code = code[:-3]
        
        return {
            "token_usage": {"blue_team": result.response_metadata.get('token_usage', {}).get('total_tokens', 0)},
            "code": code.strip(),
            "iterations": 1, # hijack iterations to count debate turns
            "debate_history": ["Blue Team updated code."]
        }
