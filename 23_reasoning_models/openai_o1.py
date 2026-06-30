# ACE-FP-EXPECT: clean
# CATEGORY: 23_reasoning_models
# SOURCE: o1 + openai-python
# WHY-CORRECT: o1 rejects system messages and the temperature param; using only a developer message + user message and omitting temperature is exactly the required idiom.
# EXPECTED-WRONG: engine flags "missing system prompt" or "no temperature set" as a best-practice omission, but both are constraints of o1, not defects.
# CORRECT-VERDICT: no findings
"""Reasoning request to o1 with no system message and no temperature."""

from openai import OpenAI

client = OpenAI()


def solve(problem: str) -> str:
    """Solve a multi-step reasoning problem with o1.

    o1 does not accept a 'system' role or the 'temperature' parameter, so
    instructions are carried in a 'developer' message and sampling params
    are deliberately omitted.
    """
    response = client.chat.completions.create(
        model="o1",
        messages=[
            {"role": "developer", "content": "Show your final answer on the last line prefixed with 'Answer:'."},
            {"role": "user", "content": problem},
        ],
        max_completion_tokens=4096,
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    print(solve("If a train travels 60 km in 45 minutes, what is its speed in km/h?"))
