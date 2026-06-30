# ACE-FP-EXPECT: clean
# CATEGORY: 33_vendor_base_urls
# SOURCE: Mistral AI, verified base_url https://api.mistral.ai/v1
# WHY-CORRECT: official OpenAI-compatible endpoint documented by Mistral AI
# EXPECTED-WRONG: analyzer flags the non-api.openai.com base_url as suspicious / SSRF / typo
# CORRECT-VERDICT: no findings
"""Call Mistral AI via its OpenAI-compatible Chat Completions API."""

import os

from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("MISTRAL_API_KEY"),
    base_url="https://api.mistral.ai/v1",
)


def main() -> None:
    response = client.chat.completions.create(
        model="mistral-large-latest",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Translate 'good morning' into French."},
        ],
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
