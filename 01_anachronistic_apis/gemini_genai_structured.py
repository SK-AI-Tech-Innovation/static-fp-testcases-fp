# ACE-FP-EXPECT: clean
# CATEGORY: 01_anachronistic_apis
# SOURCE: Google `google-genai` SDK structured output (`response_mime_type` + `response_schema`)
# WHY-CORRECT: the current `google-genai` client enforces a JSON schema via the config's
#              `response_mime_type="application/json"` and `response_schema=Model`. `response.parsed`
#              returns a validated instance — this is schema-enforced structured output.
# EXPECTED-WRONG: engine only recognizes OpenAI's `response_format`/`parse` shape and the dated
#                 `google.generativeai` package, so it flags "not structured" or rewrites toward an
#                 OpenAI call.
# CORRECT-VERDICT: no findings
"""Structured recipe extraction with the google-genai SDK and a Pydantic response schema."""
from __future__ import annotations

from google import genai
from pydantic import BaseModel

client = genai.Client()


class Recipe(BaseModel):
    name: str
    ingredients: list[str]
    minutes: int


def extract_recipe(text: str) -> Recipe:
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=text,
        config={
            "response_mime_type": "application/json",
            "response_schema": Recipe,
        },
    )
    # `parsed` is a validated Recipe instance produced under schema enforcement.
    return response.parsed
