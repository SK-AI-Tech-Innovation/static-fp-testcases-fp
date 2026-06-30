# ACE-FP-EXPECT: clean
# CATEGORY: 43_inverse_anachronism
# SOURCE: Cohere Python SDK v2 chat (`client.chat(...)`), reading text from `res.message.content[0].text`.
# WHY-CORRECT: the Cohere v2 chat response nests the reply under `message.content`, a list of content blocks;
#              the text is at `res.message.content[0].text`. This is the correct v2 access path.
# EXPECTED-WRONG: a stale engine "knows" the OpenAI shape `res.choices[0].message.content` and "fixes" the
#                 Cohere access by rewriting `res.message.content[0].text` → `res.choices[0].message.content`.
#                 The Cohere v2 response object has no `.choices` attribute → AttributeError at runtime. The
#                 "fix" breaks working code. (It would also be wrong to "fix" it to the Cohere v1
#                 `res.text` accessor, which v2 does not expose.)
# CORRECT-VERDICT: no findings — keep `res.message.content[0].text`. Do not rewrite to a `.choices[...]` or
#                  `.text` access path.
"""Cohere v2 chat reading res.message.content[0].text — not res.choices[...], by design."""
from __future__ import annotations

import os

import cohere

client = cohere.ClientV2(api_key=os.environ["COHERE_API_KEY"])


def chat(prompt: str) -> str:
    res = client.chat(
        model="command-a-03-2025",
        messages=[{"role": "user", "content": prompt}],
    )
    return res.message.content[0].text
