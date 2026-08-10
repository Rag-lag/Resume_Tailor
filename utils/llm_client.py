import os
import json
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

FREE_MODELS = [
    # "google/gemma-4-31b-it:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "openai/gpt-oss-20b:free",
    "nvidia/nemotron-nano-12b-v2-vl:free",
    # "inclusionai/ling-3.0-tiny:free",
    # "deepseek/deepseek-chat-v3.1:free",
    # "meta-llama/llama-3.3-70b-instruct:free",
    # "google/gemini-2.0-flash-exp:free",
]

def call_llm(prompt: str, system: str = "", json_mode: bool = False,
             max_retries: int = 3, temperature: float = 0.3) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    last_error = None
    for model in FREE_MODELS:
        for attempt in range(max_retries):
            try:
                kwargs = {}
                if json_mode:
                    kwargs["response_format"] = {"type": "json_object"}
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    **kwargs,
                )
                return response.choices[0].message.content
            except Exception as e:
                last_error = e
                wait = 2 ** attempt
                print(f"[{model}] attempt {attempt+1} failed: {e}. Retrying in {wait}s")
                time.sleep(wait)
        print(f"Model {model} exhausted retries, falling back to next model")
    raise RuntimeError(f"All models failed. Last error: {last_error}")


def call_llm_json(prompt: str, system: str = "", temperature: float = 0.3) -> dict:
    raw = call_llm(prompt, system=system, json_mode=True, temperature=temperature)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        fixed = call_llm(
            f"This is not valid JSON, return only corrected valid JSON:\n{raw}",
            json_mode=True, temperature=temperature,
        )
        return json.loads(fixed)