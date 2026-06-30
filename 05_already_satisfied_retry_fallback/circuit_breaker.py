# ACE-FP-EXPECT: clean
# CATEGORY: 05_already_satisfied_retry_fallback
# SOURCE: pybreaker
# WHY-CORRECT: The LLM call is wrapped in a pybreaker CircuitBreaker that trips after repeated failures and falls back to a cached/default response, so transient failures and overload are handled gracefully.
# EXPECTED-WRONG: missing resilience / missing fallback handling around the LLM call
# CORRECT-VERDICT: no findings
"""Guard the LLM call with a pybreaker circuit breaker and a safe fallback."""

import pybreaker
from openai import OpenAI

client = OpenAI(timeout=30)

llm_breaker = pybreaker.CircuitBreaker(
    fail_max=5,
    reset_timeout=60,
    exclude=[KeyboardInterrupt],
)


@llm_breaker
def _call_model(prompt: str) -> str:
    """Protected call; the breaker opens after fail_max consecutive failures."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def answer(prompt: str) -> str:
    """Return the model answer, or a fallback message when the circuit is open."""
    try:
        return _call_model(prompt)
    except pybreaker.CircuitBreakerError:
        return "The assistant is temporarily unavailable. Please try again shortly."
