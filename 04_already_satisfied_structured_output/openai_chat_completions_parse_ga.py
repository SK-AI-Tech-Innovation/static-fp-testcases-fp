# ACE-FP-EXPECT: clean
# CATEGORY: 04_already_satisfied_structured_output
# SOURCE: GA `client.chat.completions.parse(response_format=Model)` (no `.beta` namespace)
# WHY-CORRECT: the GA Structured Outputs helper enforces the Pydantic schema server-side and
#              returns a parsed instance via `message.parsed`; identical guarantee to the
#              old `client.beta.chat.completions.parse`, just promoted out of beta
# EXPECTED-WRONG: engine grounds on the `.beta.chat.completions.parse` skill example and flags
#                 the GA `.chat.completions.parse` shape as "not using structured output"
# CORRECT-VERDICT: no findings
"""Parse a math step list using the GA OpenAI Structured Outputs helper."""
from openai import OpenAI
from pydantic import BaseModel

client = OpenAI()


class Step(BaseModel):
    explanation: str
    output: str


class MathReasoning(BaseModel):
    steps: list[Step]
    final_answer: str


def solve(question: str) -> MathReasoning:
    completion = client.chat.completions.parse(
        model="gpt-4.1",
        response_format=MathReasoning,
        messages=[
            {"role": "system", "content": "Solve the problem step by step."},
            {"role": "user", "content": question},
        ],
    )
    # .parsed is a validated MathReasoning, or None only on refusal (handled by caller).
    return completion.choices[0].message.parsed
