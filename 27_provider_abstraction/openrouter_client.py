# ACE-FP-EXPECT: clean
# CATEGORY: 27_provider_abstraction
# SOURCE: OpenAI SDK pointed at OpenRouter (base_url override)
# WHY-CORRECT: OpenRouter is OpenAI-API-compatible; using the OpenAI client with base_url="https://openrouter.ai/api/v1" and a "vendor/model" id is the officially documented integration. The chat.completions.create call is valid.
# EXPECTED-WRONG: engine may flag the base_url override or the "anthropic/claude..." vendor-prefixed model as misuse of the OpenAI SDK or a wrong-provider mismatch.
"""Call models through the OpenRouter gateway using the OpenAI client."""

import os

from openai import OpenAI


def make_client() -> OpenAI:
    """Build an OpenAI-compatible client targeting OpenRouter."""
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )


def chat(model: str, prompt: str) -> str:
    """Run a chat completion against any OpenRouter-routed model."""
    client = make_client()
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
    )
    return completion.choices[0].message.content


if __name__ == "__main__":
    print(chat("anthropic/claude-sonnet-4-5", "Define eventual consistency."))
