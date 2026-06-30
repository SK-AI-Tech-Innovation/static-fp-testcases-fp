# ACE-FP-EXPECT: clean
# CATEGORY: 33_vendor_base_urls
# SOURCE: xAI Grok, verified base_url https://api.x.ai/v1
# WHY-CORRECT: official OpenAI-compatible endpoint documented by xAI
# EXPECTED-WRONG: analyzer flags the non-api.openai.com base_url as suspicious / SSRF / typo
# CORRECT-VERDICT: no findings
"""Call xAI's Grok via its OpenAI-compatible Chat Completions API."""

import os

from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("XAI_API_KEY"),
    base_url="https://api.x.ai/v1",
)


def main() -> None:
    response = client.chat.completions.create(
        model="grok-4.3",
        messages=[
            {"role": "system", "content": "You are Grok, a witty assistant."},
            {"role": "user", "content": "Explain quantum entanglement briefly."},
        ],
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
