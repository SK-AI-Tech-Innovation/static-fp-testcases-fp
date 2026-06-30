# ACE-FP-EXPECT: clean
# CATEGORY: 35_reasoning_model_constraints
# SOURCE: gpt-5.5 + openai-python (Responses API)
# WHY-CORRECT: Reasoning depth on the Responses API is controlled via reasoning={"effort": ...}, not temperature. The effort value "high" is valid; sampling params are correctly omitted.
# EXPECTED-WRONG: stale analyzer flags "missing temperature, set temperature=0" or doesn't recognize reasoning effort and suggests sampling controls; adding temperature 400s.
# CORRECT-VERDICT: no findings
"""Responses call that tunes reasoning via reasoning={"effort": "high"} instead of any sampling param."""

from openai import OpenAI

client = OpenAI()


def prove(statement: str) -> str:
    """Ask gpt-5.5 to reason hard about a statement using reasoning effort control."""
    response = client.responses.create(
        model="gpt-5.5",
        instructions="Provide a rigorous step-by-step justification.",
        input=statement,
        reasoning={"effort": "high"},
        max_output_tokens=2048,
    )
    return response.output_text


if __name__ == "__main__":
    print(prove("The sum of the first n odd numbers equals n squared."))
