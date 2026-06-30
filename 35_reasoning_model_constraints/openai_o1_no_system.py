# ACE-FP-EXPECT: clean
# CATEGORY: 35_reasoning_model_constraints
# SOURCE: o1 + openai-python (Chat Completions API)
# WHY-CORRECT: o1 does not accept the "system" role; guidance goes in a "developer" message (or a user prefix). This code uses developer + user only, which is the correct pattern. max_completion_tokens is used, no temperature.
# EXPECTED-WRONG: stale analyzer flags "missing system prompt, add a system message" and "missing temperature, set temperature=0"; adding a system role or temperature 400s on o1.
# CORRECT-VERDICT: no findings
"""o1 call with developer + user messages only (no system role), the correct o-series convention."""

from openai import OpenAI

client = OpenAI()


def review_code(snippet: str) -> str:
    """Ask o1 to review code, putting instructions in a developer message instead of system."""
    response = client.chat.completions.create(
        model="o1",
        messages=[
            {"role": "developer", "content": "You are a meticulous code reviewer. Point out real bugs only."},
            {"role": "user", "content": f"Review this code:\n\n{snippet}"},
        ],
        max_completion_tokens=4096,
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    print(review_code("def add(a, b):\n    return a - b"))
