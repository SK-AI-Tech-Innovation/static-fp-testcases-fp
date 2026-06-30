# ACE-FP-EXPECT: clean
# CATEGORY: 35_reasoning_model_constraints
# SOURCE: deepseek-reasoner + openai-python (OpenAI-compatible endpoint)
# WHY-CORRECT: deepseek-reasoner ignores temperature/top_p/penalties; the docs say not to set them. Omitting sampling params is the correct, intended usage and avoids relying on no-op fields.
# EXPECTED-WRONG: stale analyzer flags "missing temperature, set temperature=0 for determinism"; on deepseek-reasoner the param is ignored, so the "fix" is meaningless and misleading.
# CORRECT-VERDICT: no findings
"""deepseek-reasoner call via the OpenAI-compatible client, omitting sampling params it ignores."""

from openai import OpenAI

client = OpenAI(base_url="https://api.deepseek.com", api_key="sk-...")


def reason(prompt: str) -> str:
    """Get a reasoned answer from deepseek-reasoner; sampling params are intentionally not set."""
    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4096,
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    print(reason("How many distinct ways can you tile a 2x4 board with 1x2 dominoes?"))
