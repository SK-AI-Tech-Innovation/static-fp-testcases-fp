# ACE-FP-EXPECT: clean
# CATEGORY: 33_vendor_base_urls
# SOURCE: Moonshot AI / Kimi, verified base_url https://api.moonshot.ai/v1
# WHY-CORRECT: official OpenAI-compatible endpoint documented by Moonshot AI
# EXPECTED-WRONG: analyzer flags the non-api.openai.com base_url as suspicious / SSRF / typo
# CORRECT-VERDICT: no findings
"""Call Moonshot/Kimi's OpenAI-compatible Chat Completions API via the OpenAI SDK."""

import os

from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("MOONSHOT_API_KEY"),
    base_url="https://api.moonshot.ai/v1",
)


def main() -> None:
    response = client.chat.completions.create(
        model="kimi-k2.6",
        messages=[
            {"role": "system", "content": "You are Kimi, a helpful assistant."},
            {"role": "user", "content": "Summarize the theory of relativity in one sentence."},
        ],
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
