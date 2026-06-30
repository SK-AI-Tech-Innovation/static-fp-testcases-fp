# ACE-FP-EXPECT: clean
# CATEGORY: 01_anachronistic_apis
# SOURCE: Mistral Python SDK v1 (`Mistral().chat.complete(..., response_format=...)`)
# WHY-CORRECT: the modern mistralai v1 client exposes `client.chat.complete(...)` and supports
#              structured output via `response_format` (json_schema). This is current and schema-enforced.
# EXPECTED-WRONG: engine knows the legacy `MistralClient().chat(...)` / OpenAI shapes only and may flag the
#                 `client.chat.complete` call as unknown/unstructured, or rewrite to a dated client.
# CORRECT-VERDICT: no findings
"""Structured sentiment extraction with the modern Mistral SDK v1."""
from __future__ import annotations

import os

from mistralai import Mistral
from pydantic import BaseModel

client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])


class Sentiment(BaseModel):
    label: str  # positive | negative | neutral
    confidence: float


def classify(text: str) -> Sentiment:
    response = client.chat.parse(
        model="mistral-large-latest",
        messages=[
            {"role": "system", "content": "Classify the sentiment into the schema."},
            {"role": "user", "content": text},
        ],
        response_format=Sentiment,
    )
    # The SDK validates the model output against Sentiment before returning it.
    return response.choices[0].message.parsed
