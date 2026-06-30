# ACE-FP-EXPECT: clean
# CATEGORY: 43_inverse_anachronism
# SOURCE: OpenAI Responses API call against a reasoning model (`client.responses.create`).
# WHY-CORRECT: the Responses API caps output with `max_output_tokens` (NOT `max_tokens`). On a reasoning
#              model this budget covers both reasoning and visible tokens. The parameter name is correct
#              as written.
# EXPECTED-WRONG: a stale engine "knows" the cap parameter is `max_tokens` (Chat Completions era) and
#                 "fixes" `max_output_tokens` → `max_tokens`. The Responses API does not accept `max_tokens`
#                 → 400 unknown-parameter. (The reverse trap of the Chat Completions migration, where
#                 reasoning models needed `max_completion_tokens` instead of `max_tokens`.) Either way the
#                 "fix" breaks a working request.
# CORRECT-VERDICT: no findings — keep `max_output_tokens`. Do not rename it to `max_tokens`.
"""Responses API generation with an explicit output cap — max_output_tokens, by design."""
from __future__ import annotations

import os

from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def draft(prompt: str) -> str:
    resp = client.responses.create(
        model="o3",
        input=prompt,
        reasoning={"effort": "medium"},
        max_output_tokens=4000,
    )
    return resp.output_text
