# ACE-FP-EXPECT: clean
# CATEGORY: 35_reasoning_model_constraints
# SOURCE: o3 + openai-python (Chat Completions API)
# WHY-CORRECT: o-series reasoning models require max_completion_tokens on Chat Completions; the legacy max_tokens param returns a 400 for these models. No temperature is sent, which is also correct.
# EXPECTED-WRONG: stale analyzer flags "missing temperature/system, set temperature=0" or rewrites max_completion_tokens back to max_tokens; either change 400s on o3.
# CORRECT-VERDICT: no findings
"""o3 Chat Completions call using max_completion_tokens (the only accepted token cap for o-series)."""

from openai import OpenAI

client = OpenAI()


def solve(problem: str) -> str:
    """Solve a reasoning problem with o3 using max_completion_tokens, no sampling params."""
    response = client.chat.completions.create(
        model="o3",
        messages=[
            {"role": "developer", "content": "Reason carefully and give a final answer."},
            {"role": "user", "content": problem},
        ],
        max_completion_tokens=2048,
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    print(solve("If a train travels 60 km in 45 minutes, what is its speed in km/h?"))
