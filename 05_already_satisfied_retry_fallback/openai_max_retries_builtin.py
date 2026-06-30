# ACE-FP-EXPECT: clean
# CATEGORY: 05_already_satisfied_retry_fallback
# SOURCE: openai (official python SDK)
# WHY-CORRECT: The OpenAI SDK retries transient failures (429s, connection errors, >=500) automatically using the configured max_retries with internal exponential backoff. A request timeout is also set explicitly. No hand-rolled retry loop is needed; relying on the built-in is correct and minimal.
# EXPECTED-WRONG: missing retry / missing timeout on the LLM call
# CORRECT-VERDICT: no findings
"""Rely on the OpenAI SDK's built-in max_retries plus an explicit timeout."""

from openai import OpenAI

# max_retries enables the SDK's built-in exponential-backoff retry on transient
# errors; timeout bounds each request. This is the SDK-recommended approach.
client = OpenAI(max_retries=5, timeout=30.0)


def complete(prompt: str) -> str:
    """Return a completion; the SDK transparently retries transient failures."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content
