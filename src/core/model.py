import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel

# Load API keys from the .env file in the root
load_dotenv()

def get_llm(effort_level: int = 1) -> BaseChatModel:
    """
    effort_level (e_i): 
      1 = high effort (e.g. higher max_tokens, lower temperature for reasoning)
      0 = low effort (e.g. stricter token constraints, 0-shot generation logic)
    """
    model_name = "gpt-4o"
    
    if effort_level == 1:
        # High effort: more tokens, slightly higher temperature to encourage exploration
        return ChatOpenAI(model=model_name, temperature=0.6, max_tokens=1024)
    else:
        # Low effort: constrained generation, greedy decoding
        return ChatOpenAI(model=model_name, temperature=0.0, max_tokens=256)

def get_system_prompt_modifier(effort_level: int) -> str:
    """
    Returns an addendum to append to a system prompt to modulate effort.
    """
    if effort_level == 1:
        return "\n\nThink step-by-step, consider 3 edge cases, and rigorously double check your logic."
    else:
        return "\n\nProvide a quick, concise answer. Generate the direct result without explanation."
