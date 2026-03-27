import os
from dotenv import load_dotenv
from src.core.model import get_llm

if __name__ == "__main__":
    print("Testing get_llm with gpt-5.4-nano...")
    llm = get_llm(effort_level=1)
    
    print("\nSending prompt: 'Write a quick python function to calculate Fibonacci numbers.'\n")
    try:
        response = llm.invoke("Write a quick python function to calculate Fibonacci numbers.")
        print(f"Response:\n{response.content}")
        print("\n✅ Successfully connected to GPT-5.4 Nano API!")
    except Exception as e:
        print(f"❌ Error connecting to API: {e}")
