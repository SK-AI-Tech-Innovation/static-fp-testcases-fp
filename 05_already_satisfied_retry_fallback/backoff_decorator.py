# ACE-FP-EXPECT: clean
# CATEGORY: 05_already_satisfied_retry_fallback
# SOURCE: `backoff.on_exception` with exponential jittered backoff around the call
# WHY-CORRECT: the `backoff` decorator retries on the listed transient exceptions using
#              exponential backoff with jitter and a max-time ceiling — retry/backoff satisfied
# EXPECTED-WRONG: engine flags "no retry handling on network/LLM call" because the call site
#                 itself contains no loop or try/except
# CORRECT-VERDICT: no findings
"""Async embedding call guarded by a backoff exponential decorator."""
import backoff
from openai import APITimeoutError, AsyncOpenAI, RateLimitError

client = AsyncOpenAI()


@backoff.on_exception(
    backoff.expo,
    (RateLimitError, APITimeoutError),
    max_tries=6,
    max_time=60,
    jitter=backoff.full_jitter,
)
async def embed(text: str) -> list[float]:
    # backoff retries transient errors with exponential delay + jitter before giving up.
    resp = await client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return resp.data[0].embedding
