# ACE-FP-EXPECT: clean
# CATEGORY: 33_vendor_base_urls
# SOURCE: Alibaba Qwen / DashScope, verified base_url https://dashscope-intl.aliyuncs.com/compatible-mode/v1
# WHY-CORRECT: official OpenAI-compatible endpoint documented by Alibaba Cloud DashScope
# EXPECTED-WRONG: analyzer flags the non-api.openai.com base_url as suspicious / SSRF / typo
# CORRECT-VERDICT: no findings
"""Call Alibaba Qwen via DashScope's OpenAI-compatible Chat Completions API."""

import os

from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
)


def main() -> None:
    response = client.chat.completions.create(
        model="qwen3-max",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "List three uses for a paperclip."},
        ],
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
