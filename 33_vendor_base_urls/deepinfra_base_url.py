# ACE-FP-EXPECT: clean
# CATEGORY: 33_vendor_base_urls
# SOURCE: DeepInfra, verified base_url https://api.deepinfra.com/v1/openai
# WHY-CORRECT: official OpenAI-compatible endpoint documented by DeepInfra
# EXPECTED-WRONG: analyzer flags the non-api.openai.com base_url as suspicious / SSRF / typo
# CORRECT-VERDICT: no findings
"""Call DeepInfra via its OpenAI-compatible Chat Completions API."""

import os

from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("DEEPINFRA_API_KEY"),
    base_url="https://api.deepinfra.com/v1/openai",
)


def main() -> None:
    response = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-70B-Instruct",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who wrote the play Romeo and Juliet?"},
        ],
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
