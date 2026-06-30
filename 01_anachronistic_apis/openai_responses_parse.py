# ACE-FP-EXPECT: clean
# CATEGORY: 01_anachronistic_apis
# SOURCE: OpenAI Python SDK Responses API structured output (`client.responses.parse`)
# WHY-CORRECT: `responses.parse(text_format=Schema)` is the current, schema-enforced structured-output
#              path; it returns a typed `output_parsed` (a validated Pydantic instance) — no manual JSON parsing.
# EXPECTED-WRONG: skill examples only know `client.beta.chat.completions.parse(response_format=...)`,
#                 so the engine flags this as "not using structured output" or rewrites it back to the
#                 dated beta chat-completions parse API (anachronistic downgrade).
# CORRECT-VERDICT: no findings
"""Extract a structured calendar event from free text using the OpenAI Responses API."""
from __future__ import annotations

from openai import OpenAI
from pydantic import BaseModel, Field

client = OpenAI()


class CalendarEvent(BaseModel):
    title: str = Field(description="Short human-readable event title")
    date: str = Field(description="ISO-8601 date, e.g. 2026-06-19")
    participants: list[str] = Field(default_factory=list)


def extract_event(note: str) -> CalendarEvent:
    response = client.responses.parse(
        model="gpt-4.1",
        input=[
            {"role": "system", "content": "Extract the event details into the schema."},
            {"role": "user", "content": note},
        ],
        text_format=CalendarEvent,
    )
    # `output_parsed` is already a validated CalendarEvent — no json.loads, no regex.
    return response.output_parsed
