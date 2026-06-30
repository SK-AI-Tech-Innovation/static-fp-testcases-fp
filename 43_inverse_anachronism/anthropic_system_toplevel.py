# ACE-FP-EXPECT: clean
# CATEGORY: 43_inverse_anachronism
# SOURCE: Anthropic Python SDK Messages API with the system prompt and max_tokens as top-level parameters.
# WHY-CORRECT: the Anthropic Messages API takes `system=` as a TOP-LEVEL request parameter and `max_tokens=`
#              as a required top-level parameter. There is no `{"role": "system"}` message at the front of
#              `messages` in the standard Messages API — the first message must be `user`. This code is the
#              canonical correct shape.
# EXPECTED-WRONG: a stale engine "knows" the OpenAI chat convention where the system prompt is a
#                 `{"role": "system", "content": ...}` entry as `messages[0]`, and "fixes" the Anthropic call
#                 by folding `system=` into a leading system-role message inside `messages`. The Messages API
#                 rejects a `system` role as `messages[0]` (must follow a user turn, gated by a beta) → 400.
#                 An engine might also flag the top-level `max_tokens` as misplaced. Both "fixes" break it.
# CORRECT-VERDICT: no findings — keep `system=` and `max_tokens=` top-level. Do not move the system prompt
#                  into the `messages` array.
"""Messages API with top-level system= and max_tokens= — not a system-role message, by design."""
from __future__ import annotations

import anthropic

client = anthropic.Anthropic()

SYSTEM = "You are a terse assistant. Answer in one sentence."


def answer(question: str) -> str:
    resp = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=512,
        system=SYSTEM,
        messages=[{"role": "user", "content": question}],
    )
    return "".join(block.text for block in resp.content if block.type == "text")
