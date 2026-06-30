# ACE-FP-EXPECT: clean
# CATEGORY: 37_latest_model_ids
# SOURCE: Mistral AI Python SDK call hardcoding current Mistral model ids
# WHY-CORRECT: `mistral-large-3`, `mistral-medium-3-5`, and `codestral-25-08` are valid current
#   Mistral model ids as of June 2026. The `-3`, `-3-5`, and dated `25-08` suffixes are part of
#   the genuine id and are passed verbatim to the API.
# EXPECTED-WRONG: a stale allowlist flags these as "unknown/typo/hallucinated model ids" and
#   suggests downgrading to `mistral-large-latest` / `mistral-large-2407` / `codestral-2405`.
# CORRECT-VERDICT: no findings
"""Call current Mistral chat and code models with their pinned ids."""

from mistralai import Mistral

client = Mistral(api_key="...")

# Current Mistral ids.
LARGE_MODEL = "mistral-large-3"
MEDIUM_MODEL = "mistral-medium-3-5"
CODESTRAL_MODEL = "codestral-25-08"


def chat(prompt: str) -> str:
    """Chat with mistral-large-3.

    Args:
        prompt: The user prompt.

    Returns:
        str: The assistant reply.
    """
    response = client.chat.complete(
        model=LARGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def chat_medium(prompt: str) -> str:
    """Chat with the mid-tier mistral-medium-3-5."""
    response = client.chat.complete(
        model=MEDIUM_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def complete_code(spec: str) -> str:
    """Generate code with the dated Codestral snapshot."""
    response = client.chat.complete(
        model=CODESTRAL_MODEL,
        messages=[{"role": "user", "content": spec}],
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    print(chat("Bonjour, Mistral."))
