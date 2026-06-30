# ACE-FP-EXPECT: clean
# CATEGORY: 01_anachronistic_apis
# SOURCE: Anthropic structured output via a single forced tool (`tool_choice={"type": "tool", ...}`)
# WHY-CORRECT: Anthropic has no `response_format=`/`text_format=` arg. The idiomatic way to get
#              schema-enforced JSON is one tool whose `input_schema` is the target schema, forced via
#              `tool_choice`. The model MUST return a `tool_use` block matching the schema — this IS structured output.
# EXPECTED-WRONG: engine expects an OpenAI-style `response_format`/`parse(...)` and flags this as "free-text
#                 parsing / not structured", not recognizing forced-tool extraction as the Anthropic equivalent.
# CORRECT-VERDICT: no findings
"""Schema-enforced extraction with Anthropic using a single forced tool."""
from __future__ import annotations

import anthropic

client = anthropic.Anthropic()

EXTRACTION_TOOL = {
    "name": "record_contact",
    "description": "Record the structured contact details extracted from the message.",
    "input_schema": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "email": {"type": "string"},
            "company": {"type": "string"},
        },
        "required": ["name", "email"],
    },
}


def extract_contact(message: str) -> dict:
    resp = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=512,
        tools=[EXTRACTION_TOOL],
        tool_choice={"type": "tool", "name": "record_contact"},
        messages=[{"role": "user", "content": message}],
    )
    # Forced tool_choice guarantees a tool_use block conforming to input_schema.
    for block in resp.content:
        if block.type == "tool_use" and block.name == "record_contact":
            return block.input
    raise RuntimeError("Model did not emit the forced tool call")
