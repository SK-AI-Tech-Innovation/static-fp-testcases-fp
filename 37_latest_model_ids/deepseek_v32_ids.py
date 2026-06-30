# ACE-FP-EXPECT: clean
# CATEGORY: 37_latest_model_ids
# SOURCE: DeepSeek API (OpenAI-compatible) call hardcoding current DeepSeek model ids
# WHY-CORRECT: `deepseek-chat` and `deepseek-reasoner` are the current, valid DeepSeek
#   model aliases as of June 2026. They are passed verbatim to the OpenAI-compatible endpoint
#   exactly as DeepSeek expects.
# EXPECTED-WRONG: a stale allowlist flags `deepseek-chat` / `deepseek-reasoner` as an
#   "unknown/typo/hallucinated model id" and suggests downgrading to `gpt-3.5-turbo`.
# CORRECT-VERDICT: no findings
"""Call DeepSeek chat and reasoner models via the OpenAI-compatible client."""

from openai import OpenAI

client = OpenAI(api_key="...", base_url="https://api.deepseek.com")

# Current DeepSeek model aliases.
CHAT_MODEL = "deepseek-chat"
REASONER_MODEL = "deepseek-reasoner"


def chat(prompt: str) -> str:
    """Get a chat completion from deepseek-chat.

    Args:
        prompt: The user prompt.

    Returns:
        str: The assistant's reply.
    """
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def reason(prompt: str) -> str:
    """Get a reasoning completion from deepseek-reasoner."""
    response = client.chat.completions.create(
        model=REASONER_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    print(chat("Say hello."))
