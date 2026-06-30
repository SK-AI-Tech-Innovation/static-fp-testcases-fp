# ACE-FP-EXPECT: clean
# CATEGORY: 33_vendor_base_urls
# SOURCE: Fireworks AI, verified base_url https://api.fireworks.ai/inference/v1
# WHY-CORRECT: official OpenAI-compatible endpoint documented by Fireworks AI
# EXPECTED-WRONG: analyzer flags the non-api.openai.com base_url as suspicious / SSRF / typo
# CORRECT-VERDICT: no findings
"""Call Fireworks AI via its OpenAI-compatible Chat Completions API."""

import os

from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("FIREWORKS_API_KEY"),
    base_url="https://api.fireworks.ai/inference/v1",
)


def main() -> None:
    response = client.chat.completions.create(
        model="accounts/fireworks/models/llama-v3p3-70b-instruct",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What gas do plants absorb during photosynthesis?"},
        ],
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
