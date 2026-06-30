# ACE-FP-EXPECT: clean
# CATEGORY: 37_latest_model_ids
# SOURCE: xAI Grok API (OpenAI-compatible) call hardcoding current Grok model ids
# WHY-CORRECT: `grok-4.3` and `grok-4.20-0309-reasoning` are valid current xAI Grok model ids
#   as of June 2026. The dotted version and dated `-0309-reasoning` snapshot suffix are part of
#   the genuine id and are passed verbatim to the API.
# EXPECTED-WRONG: a stale allowlist flags `grok-4.3` / `grok-4.20-0309-reasoning` as an
#   "unknown/typo/hallucinated model id" (4.20 looks like a typo for 4.2) and suggests
#   downgrading to `grok-2` / `grok-beta`.
# CORRECT-VERDICT: no findings
"""Call current Grok models via the xAI OpenAI-compatible endpoint."""

from openai import OpenAI

client = OpenAI(api_key="...", base_url="https://api.x.ai/v1")

# Current Grok ids. `grok-4.20` is a genuine minor version, not a typo for 4.2.
CHAT_MODEL = "grok-4.3"
REASONING_MODEL = "grok-4.20-0309-reasoning"


def chat(prompt: str) -> str:
    """Chat with grok-4.3.

    Args:
        prompt: The user prompt.

    Returns:
        str: The assistant reply.
    """
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def reason(prompt: str) -> str:
    """Reason with the dated grok reasoning snapshot."""
    response = client.chat.completions.create(
        model=REASONING_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    print(chat("Hello, Grok."))
