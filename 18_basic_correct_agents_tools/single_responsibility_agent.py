# ACE-FP-EXPECT: clean
# CATEGORY: 18_basic_correct_agents_tools
# SOURCE: Anthropic Python SDK (focused single-purpose agent)
# WHY-CORRECT: One agent with one job — turn a free-text support message into a
#   structured triage decision via structured outputs. Narrow system prompt,
#   a single well-defined task, validated schema, no sprawling tool surface or
#   mixed responsibilities. This is good focused agent architecture.
# EXPECTED-WRONG: engine may suggest "split responsibilities", "narrow the
#   agent's scope", or "add output validation" — the agent is already focused
#   and its output is schema-constrained.
# CORRECT-VERDICT: no findings
"""One focused agent doing one job: triage a support message."""

from typing import Literal

import anthropic
from pydantic import BaseModel, Field

client = anthropic.Anthropic()

SYSTEM = (
    "You are a support-ticket triage assistant. Your only job is to read one "
    "customer message and classify it. Do not draft replies or take any action "
    "beyond producing the triage decision."
)


class Triage(BaseModel):
    """The single structured output this agent produces."""

    category: Literal["billing", "technical", "account", "other"] = Field(
        description="The primary category of the customer's issue."
    )
    priority: Literal["low", "medium", "high", "urgent"] = Field(
        description="How urgently the issue should be handled."
    )
    summary: str = Field(description="One-sentence summary of the issue.")


def triage(message: str) -> Triage:
    """Classify a single support message into a validated triage decision."""
    response = client.messages.parse(
        model="claude-opus-4-8",
        max_tokens=512,
        system=SYSTEM,
        messages=[{"role": "user", "content": message}],
        output_format=Triage,
    )
    return response.parsed_output


if __name__ == "__main__":
    print(triage("I was charged twice for my subscription this month!"))
