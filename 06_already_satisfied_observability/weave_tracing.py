# ACE-FP-EXPECT: clean
# CATEGORY: 06_already_satisfied_observability
# SOURCE: weave (Weights & Biases)
# WHY-CORRECT: weave.init() plus the @weave.op() decorator records every invocation's inputs, outputs, latency, and (for supported clients) token usage to the Weave dashboard. The LLM call is fully observable through the decorator.
# EXPECTED-WRONG: missing observability / no tracing or logging around the LLM call
# CORRECT-VERDICT: no findings
"""Trace the LLM call with a Weights & Biases Weave @weave.op() decorator."""

import weave
from openai import OpenAI

# Initializes the Weave project; all @weave.op() calls are logged to this run.
weave.init("support-assistant")

client = OpenAI()


@weave.op()
def generate_reply(question: str) -> str:
    """Decorated op: inputs, outputs, latency, and token usage are traced to Weave."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a concise support agent."},
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0].message.content
