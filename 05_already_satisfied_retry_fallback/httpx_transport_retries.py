# ACE-FP-EXPECT: clean
# CATEGORY: 05_already_satisfied_retry_fallback
# SOURCE: OpenAI SDK configured with built-in `max_retries` + an httpx transport `retries=`
# WHY-CORRECT: connection-level retries are handled by the httpx `HTTPTransport(retries=...)`
#              and request-level retries by the SDK's `max_retries`, both with explicit
#              timeouts — transient-failure resilience is configured at the client layer
# EXPECTED-WRONG: engine flags "no retry/timeout on external call" because the per-call site
#                 just does `client.chat.completions.create(...)` with no inline retry code
# CORRECT-VERDICT: no findings
"""OpenAI client wired with httpx transport retries, SDK retries, and explicit timeouts."""
import httpx
from openai import OpenAI

# Transport-level retries for connection errors; SDK-level retries for retryable statuses.
_http_client = httpx.Client(
    transport=httpx.HTTPTransport(retries=3),
    timeout=httpx.Timeout(connect=5.0, read=30.0, write=10.0, pool=5.0),
)

client = OpenAI(
    http_client=_http_client,
    max_retries=4,
    timeout=30.0,
)


def complete(prompt: str) -> str:
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content
