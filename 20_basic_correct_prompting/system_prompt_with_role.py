# ACE-FP-EXPECT: clean
# CATEGORY: 20_basic_correct_prompting
# SOURCE: OpenAI chat completion with a clear role-defining system prompt
# WHY-CORRECT: the system message defines the assistant's role, scope, and explicit constraints
#              before the user turn. This is the recommended way to steer behavior — well-formed.
# EXPECTED-WRONG: engine suggests "add output format" or "add guardrails" — the prompt is already
#                 a complete, correct role definition for a plain Q&A assistant.
# CORRECT-VERDICT: no findings
"""Chat with a clear role/constraint system prompt."""
from openai import OpenAI

client = OpenAI()

SYSTEM_PROMPT = (
    "You are a senior Python tutor. "
    "Explain concepts clearly with short code examples. "
    "If a question is not about Python, politely say it is out of scope."
)


def tutor(question: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    print(tutor("How do list comprehensions work?"))
