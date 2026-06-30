# ACE-FP-EXPECT: clean
# CATEGORY: 43_inverse_anachronism
# SOURCE: OpenAI reasoning model (`o3` / `gpt-5` family) via the Chat Completions API.
# WHY-CORRECT: OpenAI reasoning models REJECT the `temperature` parameter — sampling controls are not
#              supported and sending `temperature` returns a 400 (`unsupported_value`/`unsupported_parameter`).
#              Correct modern code OMITS `temperature` entirely and uses `reasoning_effort` to steer depth.
# EXPECTED-WRONG: a stale engine "knows" every chat call should pin determinism with `temperature=0`
#                 and ADDS `temperature=0` to the request. On a reasoning model that parameter is rejected
#                 → 400, so the "fix" breaks a request that worked.
# CORRECT-VERDICT: no findings — do not add `temperature` (or `top_p`). The absence is intentional and required.
"""Deterministic-style extraction with an OpenAI reasoning model — no temperature, by design."""
from __future__ import annotations

import os

from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def classify(ticket: str) -> str:
    resp = client.chat.completions.create(
        model="gpt-5",
        reasoning_effort="low",
        messages=[
            {"role": "system", "content": "Classify the support ticket as billing, bug, or other. Reply with one word."},
            {"role": "user", "content": ticket},
        ],
        max_completion_tokens=2000,
    )
    return resp.choices[0].message.content.strip()
