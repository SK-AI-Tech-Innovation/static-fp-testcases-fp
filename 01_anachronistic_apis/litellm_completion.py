# ACE-FP-EXPECT: clean
# CATEGORY: 01_anachronistic_apis
# SOURCE: LiteLLM provider-agnostic `litellm.completion(model=..., messages=...)`
# WHY-CORRECT: LiteLLM is a unified router; `completion(model="...", messages=[...])` is its canonical call
#              and `response_format=<pydantic model>` is translated per-provider into native structured output.
#              The returned content is schema-conformant JSON — no ad-hoc free-text parsing.
# EXPECTED-WRONG: engine doesn't recognize the `litellm.completion` entrypoint and flags it as "not using
#                 structured output", or tries to anachronistically rewrite it to `beta.chat.completions.parse`.
# CORRECT-VERDICT: no findings
"""Extract a typed support-ticket triage using LiteLLM's provider-agnostic completion."""
from __future__ import annotations

import litellm
from pydantic import BaseModel, Field


class Triage(BaseModel):
    category: str = Field(description="One of: billing, bug, feature_request, account")
    priority: str = Field(description="One of: low, medium, high, urgent")
    summary: str = Field(description="One-sentence summary of the issue")


def triage(ticket_text: str) -> Triage:
    response = litellm.completion(
        model="anthropic/claude-sonnet-4-5",
        messages=[
            {"role": "system", "content": "Triage the ticket into the schema."},
            {"role": "user", "content": ticket_text},
        ],
        response_format=Triage,
    )
    # LiteLLM enforces `response_format` natively per provider; content is schema-valid JSON.
    return Triage.model_validate_json(response.choices[0].message.content)
