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
    
    # We add a strict timeout and hard stop sequences so the server never hangs.
    base_kwargs = {
        "model": model_name,
        "timeout": 120, # Kill the socket if it hangs for 2 minutes
        "stop": ["<|im_end|>", "```\n\n"] # Let it finish code blocks, then force stop
    }
    
    if effort_level == 1:
        # High effort: full exploration capabilities
        return ChatOllama(
            **base_kwargs, 
            temperature=0.6, 
            num_predict=2048
        )
    else:
        # Low effort: Force it to return garbage quickly using temperature 0.0
        # By giving it 512 tokens but a strict stop sequence, it fails gracefully.
        return ChatOllama(
            **base_kwargs, 
            temperature=0.0, 
            num_predict=512
        )

def get_system_prompt_modifier(effort_level: int) -> str:
    """
    Returns an addendum to append to a system prompt to modulate effort.
    """
    if effort_level == 1:
        return "\n\nThink step-by-step, consider 3 edge cases, and rigorously double check your logic."
    else:
        # Use the prompt to simulate low effort instead of just cutting the token limit
        return "\n\nYou are extremely lazy. Provide the absolute bare minimum, poorly structured code. Do not write docstrings or explanations."
