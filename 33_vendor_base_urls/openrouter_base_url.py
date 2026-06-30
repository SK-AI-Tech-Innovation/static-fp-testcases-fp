# ACE-FP-EXPECT: clean
# CATEGORY: 33_vendor_base_urls
# SOURCE: OpenRouter, verified base_url https://openrouter.ai/api/v1
# WHY-CORRECT: official OpenAI-compatible endpoint documented by OpenRouter
# EXPECTED-WRONG: analyzer flags the non-api.openai.com base_url as suspicious / SSRF / typo
# CORRECT-VERDICT: no findings
"""Call OpenRouter via its OpenAI-compatible Chat Completions API."""

import os

from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)


def main() -> None:
    response = client.chat.completions.create(
        model="anthropic/claude-3.5-sonnet",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Name the largest planet in the solar system."},
        ],
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
