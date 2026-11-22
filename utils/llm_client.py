import os
import json
import time
from openai import OpenAI, NotFoundError, RateLimitError
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

FREE_MODELS = [
    # Free slugs get retired often. When all of these 404, pick new ones from
    # https://openrouter.ai/models?order=pricing-low-to-high (filter :free).
    "google/gemma-4-31b-it:free",
    "nvidia/nemotron-3.5-super-vl:free",
    "google/gemma-4-26b-a4b-it:free",
    "inclusionai/ling-3.2-flash:free",
    # Retired as of 2026-11-22:
    # "nvidia/nemotron-3-super-120b-a12b:free",
    # "openai/gpt-oss-20b:free",
    # "nvidia/nemotron-nano-12b-v2-vl:free",
    # "inclusionai/ling-3.0-tiny:free",
    # "deepseek/deepseek-chat-v3.1:free",
    # "meta-llama/llama-3.3-70b-instruct:free",
    # "google/gemini-2.0-flash-exp:free",
]

RATE_LIMIT_COOLDOWN_S = 60
_rate_limited_until: dict[str, float] = {}  # model -> time it can be tried again


def call_llm(prompt: str, system: str = "", json_mode: bool = False,
             max_retries: int = 3, temperature: float = 0.3) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    last_error = None
    # Skip models that hit a 429 recently, unless that would leave nothing to try.
    now = time.time()
    models = [m for m in FREE_MODELS if _rate_limited_until.get(m, 0) <= now] or FREE_MODELS
    for model in models:
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
            except (NotFoundError, RateLimitError) as e:
                # Retired model (404) or free-tier rate limit (429): retrying the
                # same model just burns time, so fall back immediately.
                last_error = e
                if isinstance(e, RateLimitError):
                    _rate_limited_until[model] = time.time() + RATE_LIMIT_COOLDOWN_S
                print(f"[{model}] unavailable ({e.status_code}), falling back to next model")
                break
            except Exception as e:
                last_error = e
                if attempt + 1 < max_retries:
                    wait = 2 ** attempt
                    print(f"[{model}] attempt {attempt+1} failed: {e}. Retrying in {wait}s")
                    time.sleep(wait)
        else:
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