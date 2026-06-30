# ACE-FP-EXPECT: clean
# CATEGORY: 23_reasoning_models
# SOURCE: o4-mini + openai-python
# WHY-CORRECT: reasoning models require max_completion_tokens (max_tokens is rejected); using max_completion_tokens is exactly correct.
# EXPECTED-WRONG: engine flags max_completion_tokens as a misspelling of max_tokens, or flags "max_tokens missing", but max_completion_tokens is the required param here.
# CORRECT-VERDICT: no findings
"""Reasoning request to o4-mini using max_completion_tokens (not max_tokens)."""

from openai import OpenAI

client = OpenAI()


def grade(answer: str, rubric: str) -> str:
    """Grade an answer against a rubric with o4-mini.

    Reasoning models reserve output budget for hidden reasoning tokens, so the
    output cap is set via max_completion_tokens; the legacy max_tokens param is
    not accepted on these models.
    """
    response = client.chat.completions.create(
        model="o4-mini",
        messages=[
            {"role": "developer", "content": rubric},
            {"role": "user", "content": answer},
        ],
        max_completion_tokens=2048,
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    print(grade("The mitochondria is the powerhouse of the cell.", "Score 0-10 for biological accuracy."))
