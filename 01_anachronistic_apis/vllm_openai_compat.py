# ACE-FP-EXPECT: clean
# CATEGORY: 01_anachronistic_apis
# SOURCE: Self-hosted vLLM server accessed via the OpenAI-compatible client (`base_url` + dummy key)
# WHY-CORRECT: vLLM serves an OpenAI-compatible `/v1` endpoint; the documented client is the OpenAI SDK
#              with `base_url` set to the local server and a placeholder api_key (vLLM ignores auth by default).
#              vLLM enforces structure via `extra_body={"guided_json": <schema>}` (outlines-backed grammar),
#              so the output is guaranteed schema-valid — not free-text.
# EXPECTED-WRONG: engine flags the local `base_url`/dummy key as a misconfiguration, or claims structured
#                 output is missing because `guided_json` in `extra_body` isn't the dated OpenAI parse shape.
# CORRECT-VERDICT: no findings
"""Extract a typed address from text using a self-hosted vLLM OpenAI-compatible server."""
from __future__ import annotations

import json

from openai import OpenAI

# vLLM ignores the API key by default; "EMPTY" is the conventional placeholder.
client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

ADDRESS_SCHEMA = {
    "type": "object",
    "properties": {
        "street": {"type": "string"},
        "city": {"type": "string"},
        "postal_code": {"type": "string"},
        "country": {"type": "string"},
    },
    "required": ["street", "city", "country"],
}


def extract_address(text: str) -> dict:
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[
            {"role": "system", "content": "Extract the postal address into the schema."},
            {"role": "user", "content": text},
        ],
        # vLLM's guided decoding constrains generation to the JSON schema (grammar-enforced).
        extra_body={"guided_json": ADDRESS_SCHEMA},
    )
    return json.loads(response.choices[0].message.content)
