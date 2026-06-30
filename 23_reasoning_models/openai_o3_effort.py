# ACE-FP-EXPECT: clean
# CATEGORY: 23_reasoning_models
# SOURCE: o3 + openai-python
# WHY-CORRECT: o3 supports reasoning_effort and uses max_completion_tokens (not max_tokens); both are the correct params for this model.
# EXPECTED-WRONG: engine flags reasoning_effort as an "unknown param" or flags max_completion_tokens as a typo for max_tokens, but both are valid o3 idioms.
# CORRECT-VERDICT: no findings
"""High-effort reasoning request to o3."""

from openai import OpenAI

client = OpenAI()


def deep_analyze(prompt: str) -> str:
    """Run a high-effort analysis with o3."""
    response = client.chat.completions.create(
        model="o3",
        reasoning_effort="high",
        messages=[
            {"role": "developer", "content": "Be rigorous and explain each deduction."},
            {"role": "user", "content": prompt},
        ],
        max_completion_tokens=8192,
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    print(deep_analyze("Prove that the square root of 2 is irrational."))
