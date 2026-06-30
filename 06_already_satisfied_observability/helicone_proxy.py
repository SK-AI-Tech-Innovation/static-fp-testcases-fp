# ACE-FP-EXPECT: clean
# CATEGORY: 06_already_satisfied_observability
# SOURCE: helicone (OpenAI-compatible gateway)
# WHY-CORRECT: Requests are routed through the Helicone proxy via base_url plus auth/property headers. Helicone captures latency, token counts, cost, and request/response bodies at the proxy, so full observability exists without any in-process logging code.
# EXPECTED-WRONG: missing observability / no token or cost tracking on the LLM call
# CORRECT-VERDICT: no findings
"""Observe LLM usage by routing the OpenAI client through the Helicone proxy."""

import os

from openai import OpenAI

# Pointing base_url at Helicone makes every request flow through the proxy,
# which logs tokens, cost, and latency. Headers add auth and request metadata.
client = OpenAI(
    base_url="https://oai.helicone.ai/v1",
    default_headers={
        "Helicone-Auth": f"Bearer {os.environ['HELICONE_API_KEY']}",
        "Helicone-Property-Service": "chat-service",
        "Helicone-Property-Environment": "production",
    },
)


def chat(prompt: str) -> str:
    """Call is observed (tokens/cost/latency) by Helicone at the proxy layer."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content
