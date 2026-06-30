# ACE-FP-EXPECT: clean
# CATEGORY: 01_anachronistic_apis
# SOURCE: Ollama Python client structured outputs (`chat(..., format=Schema.model_json_schema())`)
# WHY-CORRECT: Ollama enforces a JSON schema when `format=` is a JSON-schema dict; passing
#              `Model.model_json_schema()` constrains decoding, and `Model.model_validate_json` validates it.
#              This is genuine schema-enforced structured output for a local model.
# EXPECTED-WRONG: engine only recognizes OpenAI/`response_format`; it flags the `format=` schema dict as not
#                 structured output, or treats `response.message.content` as unparsed free text needing regex.
# CORRECT-VERDICT: no findings
"""Structured country-info extraction from a local model via Ollama's `format` schema."""
from __future__ import annotations

from ollama import chat
from pydantic import BaseModel


class Country(BaseModel):
    name: str
    capital: str
    languages: list[str]


def describe_country(name: str) -> Country:
    response = chat(
        model="llama3.1",
        messages=[{"role": "user", "content": f"Tell me about {name}."}],
        format=Country.model_json_schema(),
    )
    # Decoding was constrained to the schema; validate to get a typed instance.
    return Country.model_validate_json(response.message.content)
