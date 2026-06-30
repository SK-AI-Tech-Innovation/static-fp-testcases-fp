# ACE-FP-EXPECT: 1
# CATEGORY: 12_severity_miscalibration
# SOURCE: a correct Anthropic call that works fine but emits no tracing/logging of requests or latency
# WHY-CORRECT: the LLM call is correct (explicit model, max_tokens, system, parsed text). The only gap is
#              observability — there is no logging/tracing around the call. Per the rubric, a missing
#              observability/logging concern is NEVER critical; at most it is a single suggestion to add
#              request/latency logging or tracing.
# EXPECTED-WRONG: escalating "no logging/observability" to `critical`, or emitting multiple findings.
# CORRECT-VERDICT: at most 1 suggestion ("add tracing/logging around the model call"), NEVER critical
"""A correct Claude classification call with no logging/tracing (observability gap only)."""
from __future__ import annotations

import anthropic

client = anthropic.Anthropic()


def classify_sentiment(text: str) -> str:
    # Works correctly; simply emits no logs/spans for the request or its latency.
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=64,
        system="Classify the sentiment as exactly one of: positive, negative, neutral.",
        messages=[{"role": "user", "content": text}],
    )
    return "".join(b.text for b in response.content if b.type == "text").strip().lower()


if __name__ == "__main__":
    print(classify_sentiment("The product arrived early and works great."))
