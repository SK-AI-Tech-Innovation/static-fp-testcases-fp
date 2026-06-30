# ACE-FP-EXPECT: scoped-out
# CATEGORY: 03_out_of_scope_security
# SOURCE: redacting emails / phone / SSN out of user text before it is placed in an Anthropic prompt
# WHY-CORRECT: PII handling is a SECURITY concern, delegated to the separate Sentinel engine. The static
#              LLM-pattern engine must stay silent on security regardless of how good or bad the redaction
#              is. Here the redaction is in fact correct (sent text is scrubbed), so even Sentinel would pass.
# EXPECTED-WRONG: a security/PII finding ("PII may reach the model", "validate redaction", "data leakage")
#                 — security is out-of-scope; static must defer to Sentinel and emit nothing.
# CORRECT-VERDICT: no findings (security/PII is Sentinel's domain; static stays silent)
"""Scrub PII from user input before prompting Claude (security-correct; static must defer to Sentinel)."""
from __future__ import annotations

import re

import anthropic

client = anthropic.Anthropic()

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE = re.compile(r"\b\d{3}[-.]?\d{3,4}[-.]?\d{4}\b")
_SSN = re.compile(r"\b\d{6}[-]?\d{7}\b")


def redact_pii(text: str) -> str:
    text = _EMAIL.sub("[EMAIL]", text)
    text = _PHONE.sub("[PHONE]", text)
    text = _SSN.sub("[SSN]", text)
    return text


def classify_ticket(raw_user_text: str) -> str:
    safe_text = redact_pii(raw_user_text)
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=256,
        system="Classify this support ticket as one of: billing, technical, account.",
        messages=[{"role": "user", "content": safe_text}],
    )
    return "".join(b.text for b in response.content if b.type == "text").strip()


if __name__ == "__main__":
    print(classify_ticket("Hi, I'm jane@example.com / 010-1234-5678, my login is broken."))
