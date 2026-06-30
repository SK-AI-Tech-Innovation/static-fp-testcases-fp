# ACE-FP-EXPECT: clean
# CATEGORY: 33_vendor_base_urls
# SOURCE: self-hosted vLLM OpenAI-compatible server, verified base_url http://localhost:8000/v1
# WHY-CORRECT: vLLM exposes an OpenAI-compatible server; api_key="EMPTY" is the documented placeholder
# EXPECTED-WRONG: analyzer flags localhost / http base_url and "EMPTY" key as SSRF / suspicious
# CORRECT-VERDICT: no findings
"""Call a locally self-hosted vLLM OpenAI-compatible server via the OpenAI SDK."""

from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
)


def main() -> None:
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Give me a fun fact about octopuses."},
        ],
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
