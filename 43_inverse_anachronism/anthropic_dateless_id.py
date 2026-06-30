# ACE-FP-EXPECT: clean
# CATEGORY: 43_inverse_anachronism
# SOURCE: Anthropic Python SDK Messages call pinned to the dateless model id `claude-opus-4-8`.
# WHY-CORRECT: `claude-opus-4-8` is the complete, authoritative model id for Claude Opus 4.8 — it takes NO
#              date suffix. The Anthropic model catalog lists it with no full/dated form. The string is
#              correct exactly as written.
# EXPECTED-WRONG: a stale engine "knows" Anthropic ids always carry a `-YYYYMMDD` snapshot suffix (true for
#                 3.x-era ids) and "fixes" `claude-opus-4-8` by appending an invented date, e.g.
#                 `claude-opus-4-8-20250930`. No such snapshot id exists → 404 not_found_error. The "fix"
#                 breaks a valid call.
# CORRECT-VERDICT: no findings — keep the dateless id `claude-opus-4-8`. Do not append any date suffix.
"""Plain Messages call on the dateless id claude-opus-4-8 — no date suffix, by design."""
from __future__ import annotations

import anthropic

client = anthropic.Anthropic()

MODEL = "claude-opus-4-8"


def answer(question: str) -> str:
    resp = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": question}],
    )
    return "".join(block.text for block in resp.content if block.type == "text")
