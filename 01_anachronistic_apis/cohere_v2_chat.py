# ACE-FP-EXPECT: clean
# CATEGORY: 01_anachronistic_apis
# SOURCE: Cohere Python SDK v2 (`cohere.ClientV2().chat(...)`) with JSON-schema response_format
# WHY-CORRECT: Cohere v2 uses its own message/content shape (list-of-content-objects, `messages=`) and
#              enforces structure via `response_format={"type": "json_object", "json_schema": ...}`.
#              The schema IS enforced server-side; `message.content[0].text` is guaranteed valid JSON.
# EXPECTED-WRONG: skill examples only know OpenAI's `beta.chat.completions.parse(response_format=Pydantic)`,
#                 so the engine flags Cohere's different message shape / `client.chat()` as "not structured"
#                 or rewrites it into an anachronistic OpenAI chat-completions parse call.
# CORRECT-VERDICT: no findings
"""Extract a typed invoice summary from text using the Cohere v2 chat API."""
from __future__ import annotations

import json

import cohere

client = cohere.ClientV2()

INVOICE_SCHEMA = {
    "type": "object",
    "properties": {
        "invoice_number": {"type": "string"},
        "total_amount": {"type": "number"},
        "currency": {"type": "string"},
        "line_items": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["invoice_number", "total_amount", "currency"],
}


def extract_invoice(raw_text: str) -> dict:
    response = client.chat(
        model="command-r-plus-08-2024",
        messages=[
            {
                "role": "system",
                "content": "Extract the invoice fields into the provided schema.",
            },
            {"role": "user", "content": raw_text},
        ],
        response_format={"type": "json_object", "json_schema": INVOICE_SCHEMA},
    )
    # response_format with a json_schema guarantees schema-valid JSON in the text block.
    return json.loads(response.message.content[0].text)
