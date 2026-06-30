# ACE-FP-EXPECT: clean
# CATEGORY: 43_inverse_anachronism
# SOURCE: OpenAI Responses API structured output via `client.responses.parse(text_format=Model)`.
# WHY-CORRECT: the Responses API is OpenAI's current surface; `responses.parse(..., text_format=PydanticModel)`
#              returns a parsed instance on `.output_parsed`. This is the modern, correct way to get typed
#              structured output.
# EXPECTED-WRONG: a stale engine "knows" structured parsing lives at `client.beta.chat.completions.parse(
#                 response_format=Model)` and "downgrades" the call to that legacy Chat Completions beta —
#                 swapping `responses.parse`→`beta.chat.completions.parse`, `text_format`→`response_format`,
#                 and `input`→`messages`, and reading `.choices[0].message.parsed` instead of `.output_parsed`.
#                 That abandons the Responses API entirely and breaks reasoning-item / state continuity.
# CORRECT-VERDICT: no findings — do not rewrite to the `beta.chat.completions` shape. The Responses API call
#                  is the correct modern form.
"""Typed structured output via the OpenAI Responses API — responses.parse, by design."""
from __future__ import annotations

import os

from openai import OpenAI
from pydantic import BaseModel

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


class Contact(BaseModel):
    name: str
    email: str
    company: str


def extract_contact(blurb: str) -> Contact:
    resp = client.responses.parse(
        model="gpt-5",
        input=[
            {"role": "system", "content": "Extract the contact fields from the text."},
            {"role": "user", "content": blurb},
        ],
        text_format=Contact,
    )
    return resp.output_parsed
