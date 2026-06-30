# ACE-FP-EXPECT: scoped-out
# CATEGORY: 03_out_of_scope_security
# SOURCE: validating/guarding model output for unsafe content before returning it to the caller
# WHY-CORRECT: output guardrails (blocking unsafe/disallowed content) are a SECURITY concern owned by the
#              Sentinel engine. Static must stay silent on guardrails. The code is also security-correct:
#              the model's response is screened against a blocklist and a safe fallback is returned if it
#              trips, so unsafe content never reaches the caller.
# EXPECTED-WRONG: a guardrail/safety finding ("add output moderation", "unsafe content may be returned",
#                 "missing content filter") — security is out-of-scope; static must defer to Sentinel.
# CORRECT-VERDICT: no findings (output guardrails are Sentinel's domain; static stays silent)
"""Screen Claude output through a safety guardrail before returning (security-correct; defer to Sentinel)."""
from __future__ import annotations

import anthropic

client = anthropic.Anthropic()

_BLOCKED_TOPICS = ("weapon synthesis", "self-harm instructions", "malware payload")
_SAFE_FALLBACK = "I can't help with that request."


def passes_guardrail(text: str) -> bool:
    lowered = text.lower()
    return not any(topic in lowered for topic in _BLOCKED_TOPICS)


def generate_safe_reply(user_prompt: str) -> str:
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=600,
        messages=[{"role": "user", "content": user_prompt}],
    )
    candidate = "".join(b.text for b in response.content if b.type == "text").strip()
    if not passes_guardrail(candidate):
        return _SAFE_FALLBACK
    return candidate


if __name__ == "__main__":
    print(generate_safe_reply("Explain how photosynthesis works."))
