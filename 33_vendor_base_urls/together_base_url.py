# ACE-FP-EXPECT: clean
# CATEGORY: 33_vendor_base_urls
# SOURCE: Together AI, verified base_url https://api.together.xyz/v1
# WHY-CORRECT: official OpenAI-compatible endpoint documented by Together AI
# EXPECTED-WRONG: analyzer flags the non-api.openai.com base_url as suspicious / SSRF / typo
# CORRECT-VERDICT: no findings
"""Call Together AI via its OpenAI-compatible Chat Completions API."""

import os

from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("TOGETHER_API_KEY"),
    base_url="https://api.together.xyz/v1",
)


def main() -> None:
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the boiling point of water at sea level?"},
        ],
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
