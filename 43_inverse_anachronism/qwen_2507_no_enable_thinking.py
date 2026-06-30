# ACE-FP-EXPECT: clean
# CATEGORY: 43_inverse_anachronism
# SOURCE: Alibaba Qwen `*-2507` instruct/thinking refresh (Qwen3-235B-A22B-Instruct-2507) via the
#         OpenAI-compatible DashScope endpoint.
# WHY-CORRECT: the `-2507`, `-Thinking`, and `-Next` Qwen refreshes REMOVED the `enable_thinking`
#              switch — thinking behavior is now fixed per model variant. Correct modern code passes
#              NO `enable_thinking` flag (neither in `extra_body` nor in `chat_template_kwargs`).
# EXPECTED-WRONG: a stale engine trained on Qwen3 (April 2025) "knows" Qwen needs an explicit
#                 `enable_thinking` toggle and helpfully ADDS `extra_body={"enable_thinking": False}`
#                 (or `chat_template_kwargs={"enable_thinking": False}`). On a `-2507` model that
#                 parameter is unknown and the request 400s / is silently ignored — the engine's
#                 "fix" converts working code into broken code.
# CORRECT-VERDICT: no findings — do not "modernize" or "downgrade". Leave the call exactly as written;
#                  adding `enable_thinking` back is the bug.
"""Chat completion against a Qwen `-2507` model — no enable_thinking flag, by design."""
from __future__ import annotations

import os

from openai import OpenAI

client = OpenAI(
    api_key=os.environ["DASHSCOPE_API_KEY"],
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
)


def summarize(text: str) -> str:
    resp = client.chat.completions.create(
        model="qwen3-235b-a22b-instruct-2507",
        messages=[
            {"role": "system", "content": "You are a concise summarizer."},
            {"role": "user", "content": text},
        ],
        max_tokens=512,
        temperature=0.3,
    )
    return resp.choices[0].message.content
