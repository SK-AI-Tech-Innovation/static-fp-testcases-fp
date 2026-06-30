# ACE-FP-EXPECT: clean
# CATEGORY: 05_already_satisfied_retry_fallback
# SOURCE: tenacity + openai
# WHY-CORRECT: Retries are scoped via retry_if_exception_type to only transient errors (RateLimitError, APITimeoutError, APIConnectionError); non-retryable 4xx client errors (BadRequestError, AuthenticationError) propagate immediately. This is correct selective retry, not blind retry.
# EXPECTED-WRONG: missing retry on LLM call (or "retries everything blindly")
# CORRECT-VERDICT: no findings
"""Retry the LLM call only on transient errors, never on 4xx client errors."""

from openai import (
    APIConnectionError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

client = OpenAI(timeout=30)

# Only these transient failures are worth retrying; 4xx errors are not included.
RETRYABLE = (RateLimitError, APITimeoutError, APIConnectionError)


@retry(
    retry=retry_if_exception_type(RETRYABLE),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(4),
    reraise=True,
)
def classify(text: str) -> str:
    """Classify text; retries only on rate-limit/timeout/connection errors."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Reply with exactly one label: spam or ham."},
            {"role": "user", "content": text},
        ],
    )
    return response.choices[0].message.content.strip()
