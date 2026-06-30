# ACE-FP-EXPECT: clean
# CATEGORY: 23_reasoning_models
# SOURCE: o1-mini + openai-python
# WHY-CORRECT: o1-mini does not support the system role; carrying instructions in the user turn and using max_completion_tokens is the correct idiom.
# EXPECTED-WRONG: engine flags "no system prompt" as a best-practice omission, but o1-mini rejects system messages by design.
# CORRECT-VERDICT: no findings
"""Reasoning request to o1-mini with instructions folded into the user turn."""

from openai import OpenAI

client = OpenAI()


def fix_bug(code: str) -> str:
    """Ask o1-mini to find and fix a bug.

    o1-mini does not accept a 'system' role, so the instruction is prepended
    to the user message instead.
    """
    instruction = "Find the bug and return only the corrected function.\n\n"
    response = client.chat.completions.create(
        model="o1-mini",
        messages=[{"role": "user", "content": instruction + code}],
        max_completion_tokens=4096,
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    print(fix_bug("def add(a, b):\n    return a - b\n"))
