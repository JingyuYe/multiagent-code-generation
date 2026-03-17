import os
from langchain_ollama import ChatOllama
from langchain_core.language_models.chat_models import BaseChatModel

def get_llm(effort_level: int = 1) -> BaseChatModel:
    """
    effort_level (e_i): 
      1 = high effort (e.g. higher max_tokens, lower temperature for reasoning)
      0 = low effort (e.g. stricter token constraints, 0-shot generation logic)
      
    Runs locally using Ollama. For a 16GB machine, qwen2.5-coder:7b is the 
    absolute best open-source LLM for code generation in the ~4-8GB memory envelope,
    scoring aggressively high on HumanEval/MBPP without risking memory page-outs.
    """
    model_name = "qwen2.5-coder:7b"
    
    if effort_level == 1:
        # High effort: more tokens, slightly higher temperature to encourage exploration
        return ChatOllama(model=model_name, temperature=0.6, num_predict=2048)
    else:
        # Low effort: constrained generation, greedy decoding
        return ChatOllama(model=model_name, temperature=0.0, num_predict=256)

def get_system_prompt_modifier(effort_level: int) -> str:
    """
    Returns an addendum to append to a system prompt to modulate effort.
    """
    if effort_level == 1:
        return "\n\nThink step-by-step, consider 3 edge cases, and rigorously double check your logic."
    else:
        return "\n\nProvide a quick, concise answer. Generate the direct result without explanation."
