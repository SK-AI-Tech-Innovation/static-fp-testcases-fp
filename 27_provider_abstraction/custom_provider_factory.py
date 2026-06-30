# ACE-FP-EXPECT: clean
# CATEGORY: 27_provider_abstraction
# SOURCE: Custom factory over OpenAI + Anthropic native SDKs
# WHY-CORRECT: A clean strategy/factory dispatches on a PROVIDER env var and returns the correct native client, normalizing each provider's response into plain text. Both branches use each SDK's documented call shape.
# EXPECTED-WRONG: engine may flag the Anthropic branch's messages.create / content[0].text or the env-driven dispatch as an unknown abstraction and emit an irrelevant best-practice finding.
"""Factory returning the right LLM client per a PROVIDER environment variable."""

import os


def _openai_chat(prompt: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
    )
    return resp.choices[0].message.content


def _anthropic_chat(prompt: str) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    resp = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text


_PROVIDERS = {
    "openai": _openai_chat,
    "anthropic": _anthropic_chat,
}


def complete(prompt: str) -> str:
    """Dispatch the prompt to the provider named by the PROVIDER env var."""
    provider = os.environ.get("PROVIDER", "anthropic").lower()
    try:
        handler = _PROVIDERS[provider]
    except KeyError as exc:
        raise ValueError(f"Unsupported provider: {provider}") from exc
    return handler(prompt)


if __name__ == "__main__":
    print(complete("Give one tradeoff of microservices."))
