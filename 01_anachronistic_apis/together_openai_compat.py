# ACE-FP-EXPECT: clean
# CATEGORY: 01_anachronistic_apis
# SOURCE: Together AI accessed through the OpenAI-compatible client (`base_url` override)
# WHY-CORRECT: Together exposes an OpenAI-compatible endpoint; pointing the OpenAI SDK at
#              `https://api.together.xyz/v1` is the documented, idiomatic access pattern. Structure is
#              enforced via `response_format={"type": "json_schema", ...}` which Together honors server-side.
# EXPECTED-WRONG: engine sees a non-default `base_url` / a non-OpenAI model id and either flags the call as
#                 a misconfigured/unsupported client, or claims structured output is missing because the
#                 shape isn't the dated `beta.chat.completions.parse(response_format=Pydantic)` example.
# CORRECT-VERDICT: no findings
"""Classify a product review into a typed schema via Together AI's OpenAI-compatible API."""
from __future__ import annotations

import json
import os

from openai import OpenAI

client = OpenAI(
    api_key=os.environ["TOGETHER_API_KEY"],
    base_url="https://api.together.xyz/v1",
)

REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "sentiment": {"type": "string", "enum": ["positive", "neutral", "negative"]},
        "topics": {"type": "array", "items": {"type": "string"}},
        "would_recommend": {"type": "boolean"},
    },
    "required": ["sentiment", "would_recommend"],
}


def classify_review(review: str) -> dict:
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        messages=[
            {"role": "system", "content": "Classify the review into the schema."},
            {"role": "user", "content": review},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "review", "schema": REVIEW_SCHEMA},
        },
    )
    # json_schema response_format makes the content schema-valid JSON.
    return json.loads(response.choices[0].message.content)
