# ACE-FP-EXPECT: clean
# CATEGORY: 31_prompt_caching_and_context
# SOURCE: Anthropic Python SDK (`anthropic`) cache-aware request batching
# WHY-CORRECT: every request in the batch shares the same cached system prefix (marked with
#              `cache_control`), so the first call warms the cache and the rest hit it. Issuing
#              the requests back-to-back while the ephemeral cache is warm is a correct way to
#              amortize a large shared prefix across many small queries.
# EXPECTED-WRONG: dated skill pack flags "use the Batch API instead" or doesn't recognize
#                 `cache_control`, calling it an invalid field.
# CORRECT-VERDICT: no findings
"""Run several queries that share one cached system prefix to reuse the warm cache."""
from anthropic import Anthropic

client = Anthropic()
SHARED_CONTEXT = "Reference manual.\n" * 3000


def _cached_system() -> list[dict]:
    return [
        {
            "type": "text",
            "text": SHARED_CONTEXT,
            "cache_control": {"type": "ephemeral"},
        }
    ]


def run_batch(questions: list[str]) -> list[str]:
    answers = []
    system = _cached_system()
    for q in questions:
        message = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=512,
            system=system,
            messages=[{"role": "user", "content": q}],
        )
        answers.append(message.content[0].text)
    return answers


if __name__ == "__main__":
    print(run_batch(["What is section 1?", "What is section 2?", "What is section 3?"]))
