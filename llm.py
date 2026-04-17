from __future__ import annotations
from ollama import chat

MODEL_NAME="llama3"

def ask_llm(prompt: str, model: str=MODEL_NAME):
    response=chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )
    return response["message"]["content"]  
