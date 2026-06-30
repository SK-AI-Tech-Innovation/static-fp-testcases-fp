# ACE-FP-EXPECT: clean
# CATEGORY: 23_reasoning_models
# SOURCE: o3 + openai-python (Responses API, structured output)
# WHY-CORRECT: responses.parse with reasoning={"effort": ...} and a Pydantic text_format is the correct way to get schema-validated output from a reasoning model.
# EXPECTED-WRONG: engine flags reasoning param + text_format together as unsupported, or flags the absence of temperature, but this is the documented Responses API idiom.
# CORRECT-VERDICT: no findings
"""Structured output from o3 via the Responses API with reasoning effort."""

from openai import OpenAI
from pydantic import BaseModel

client = OpenAI()


class Triage(BaseModel):
    """Structured triage result."""

    severity: str
    summary: str
    needs_human: bool


def triage(ticket: str) -> Triage:
    """Triage a support ticket into a structured result with o3."""
    response = client.responses.parse(
        model="o3",
        reasoning={"effort": "high"},
        input=[
            {"role": "developer", "content": "Classify the ticket. severity is one of low/medium/high."},
            {"role": "user", "content": ticket},
        ],
        text_format=Triage,
    )
    return response.output_parsed


if __name__ == "__main__":
    result = triage("The production database is down and customers cannot check out.")
    print(result.severity, result.needs_human)
