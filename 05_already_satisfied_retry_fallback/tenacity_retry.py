# ACE-FP-EXPECT: clean
# CATEGORY: 05_already_satisfied_retry_fallback
# SOURCE: `tenacity.retry` with exponential backoff around an LLM call
# WHY-CORRECT: transient failures (rate limits, timeouts, connection errors) are retried with
#              bounded exponential backoff and a stop condition — the resilience principle is
#              fully satisfied by the decorator
# EXPECTED-WRONG: engine flags "missing retry/backoff for external API call" because the body
#                 of the function has a single bare `client.chat.completions.create(...)`
# CORRECT-VERDICT: no findings
"""Call the chat API with tenacity exponential-backoff retries on transient errors."""
from openai import APIConnectionError, OpenAI, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

client = OpenAI()


@retry(
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(5),
    reraise=True,
)
def complete(prompt: str) -> str:
    # Any RateLimitError/APIConnectionError here is retried with backoff by the decorator.
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content
