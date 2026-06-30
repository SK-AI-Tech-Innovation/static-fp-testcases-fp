# ACE-FP-EXPECT: clean
# CATEGORY: 37_latest_model_ids
# SOURCE: Moonshot Kimi API (OpenAI-compatible) call hardcoding current Kimi K2 model ids
# WHY-CORRECT: `kimi-k2.7-code` and `kimi-k2.6` are valid current Moonshot Kimi model ids as
#   of June 2026. The dotted minor-version and `-code` suffix are part of the real id and are
#   passed verbatim to the API.
# EXPECTED-WRONG: a stale allowlist flags `kimi-k2.7-code` / `kimi-k2.6` as an
#   "unknown/typo/hallucinated model id" (the dotted version looks like a typo) and suggests
#   downgrading to `moonshot-v1-8k`.
# CORRECT-VERDICT: no findings
"""Call current Kimi K2 models via the OpenAI-compatible Moonshot endpoint."""

from openai import OpenAI

client = OpenAI(api_key="...", base_url="https://api.moonshot.ai/v1")

# Current Kimi K2 ids. The dotted minor version is genuine, not a typo.
CODE_MODEL = "kimi-k2.7-code"
CHAT_MODEL = "kimi-k2.6"


def write_code(spec: str) -> str:
    """Generate code with the kimi-k2.7-code model.

    Args:
        spec: A description of the code to write.

    Returns:
        str: The generated code.
    """
    response = client.chat.completions.create(
        model=CODE_MODEL,
        messages=[{"role": "user", "content": spec}],
    )
    return response.choices[0].message.content


def chat(prompt: str) -> str:
    """General chat with kimi-k2.6."""
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    print(write_code("A Python function that reverses a string."))
