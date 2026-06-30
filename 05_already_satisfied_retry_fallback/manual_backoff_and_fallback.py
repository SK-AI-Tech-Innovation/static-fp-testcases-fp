# ACE-FP-EXPECT: clean
# CATEGORY: 05_already_satisfied_retry_fallback
# SOURCE: hand-rolled try/except loop with exponential backoff + secondary-model fallback
# WHY-CORRECT: transient errors are retried with growing sleep intervals; once the primary
#              model is exhausted, a different fallback model is tried — full retry+fallback
# EXPECTED-WRONG: engine flags "missing retry/backoff" because there is no recognized decorator
#                 (tenacity/backoff) — it does not credit the explicit loop implementation
# CORRECT-VERDICT: no findings
"""Manual exponential-backoff retry loop with a fallback model for chat completions."""
import random
import time

from openai import APIConnectionError, OpenAI, RateLimitError

client = OpenAI()
_PRIMARY = "gpt-4.1"
_FALLBACK = "gpt-4.1-mini"
_TRANSIENT = (RateLimitError, APIConnectionError)


def _call(model: str, prompt: str) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content


def complete(prompt: str, max_attempts: int = 5) -> str:
    for model in (_PRIMARY, _FALLBACK):
        delay = 1.0
        for attempt in range(max_attempts):
            try:
                return _call(model, prompt)
            except _TRANSIENT:
                if attempt == max_attempts - 1:
                    break  # exhausted this model -> fall over to the next model
                time.sleep(delay + random.uniform(0, 0.5))  # exponential backoff + jitter
                delay *= 2
    raise RuntimeError("All models exhausted after retries")
