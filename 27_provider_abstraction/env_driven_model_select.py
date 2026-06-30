# ACE-FP-EXPECT: clean
# CATEGORY: 27_provider_abstraction
# SOURCE: Config/env-driven model selection over the OpenAI SDK
# WHY-CORRECT: Reading model id, base_url, and api_key from a config dataclass populated by env vars is a clean, intentional way to swap providers without code changes. The resulting chat.completions.create call is standard.
# EXPECTED-WRONG: engine may flag the indirect/env-resolved model string (never a literal) or the configurable base_url as a misconfiguration or unknown-model finding.
"""Select model and provider endpoint cleanly from environment configuration."""

import os
from dataclasses import dataclass

from openai import OpenAI


@dataclass(frozen=True)
class ModelConfig:
    """Resolved provider configuration."""

    model: str
    base_url: str
    api_key: str


def load_config() -> ModelConfig:
    """Build a ModelConfig from environment variables with sane defaults."""
    return ModelConfig(
        model=os.environ.get("LLM_MODEL", "gpt-4o"),
        base_url=os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1"),
        api_key=os.environ["LLM_API_KEY"],
    )


def chat(prompt: str) -> str:
    """Run a completion using the environment-resolved configuration."""
    cfg = load_config()
    client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
    response = client.chat.completions.create(
        model=cfg.model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    print(chat("What is backpressure in streaming systems?"))
