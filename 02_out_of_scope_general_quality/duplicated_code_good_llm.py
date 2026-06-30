# ACE-FP-EXPECT: scoped-out
# CATEGORY: 02_out_of_scope_general_quality
# SOURCE: two correct Anthropic calls (summarize + translate) with near-duplicate setup blocks
# WHY-CORRECT: both calls are correct LLM usage — explicit model, max_tokens, system prompt, text parsing.
#              The two functions share near-identical request scaffolding (a DRY violation), but DRY /
#              de-duplication is a general code-quality concern that is out-of-scope for static here.
#              (Per repo guidance, explicit-over-DRY is even an accepted style.)
# EXPECTED-WRONG: "duplicated code / extract a shared helper / violates DRY" refactoring findings.
# CORRECT-VERDICT: no findings (duplication is out-of-scope; both LLM calls are already correct)
"""Summarize and translate via two correct, near-duplicate Anthropic calls (intentional duplication)."""
from __future__ import annotations

import anthropic

client = anthropic.Anthropic()


def summarize(text: str) -> str:
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=512,
        system="You are a concise summarizer. Produce a 2-sentence summary.",
        messages=[{"role": "user", "content": text}],
    )
    out = []
    for block in response.content:
        if block.type == "text":
            out.append(block.text)
    return "".join(out).strip()


def translate_to_korean(text: str) -> str:
    # Intentionally duplicates the scaffolding above instead of sharing a helper.
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=512,
        system="You are a translator. Translate the user's text into natural Korean.",
        messages=[{"role": "user", "content": text}],
    )
    out = []
    for block in response.content:
        if block.type == "text":
            out.append(block.text)
    return "".join(out).strip()


if __name__ == "__main__":
    article = "Large language models are increasingly used for document workflows."
    print(summarize(article))
    print(translate_to_korean(article))
