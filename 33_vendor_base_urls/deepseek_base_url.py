# ACE-FP-EXPECT: clean
# CATEGORY: 33_vendor_base_urls
# SOURCE: DeepSeek, verified base_url https://api.deepseek.com
# WHY-CORRECT: official OpenAI-compatible endpoint documented by DeepSeek
# EXPECTED-WRONG: analyzer flags the non-api.openai.com base_url as suspicious / SSRF / typo
# CORRECT-VERDICT: no findings
"""Call DeepSeek's OpenAI-compatible Chat Completions API via the OpenAI SDK."""

import os

from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)


def main() -> None:
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
        ],
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
