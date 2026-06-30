# ACE-FP-EXPECT: clean
# CATEGORY: 15_correct_terse_code
# SOURCE: A minimal, correct single LLM call
# WHY-CORRECT: Short by design and fully correct: explicit model, max_tokens set, clear message. Nothing is missing for what it does.
# EXPECTED-WRONG: Engine inflates suggestions (add retries, logging, temperature, system prompt, type hints elsewhere) on code that is already correct and complete
# CORRECT-VERDICT: no findings
"""A minimal correct completion call."""

from anthropic import Anthropic

client = Anthropic()


def ask(question: str) -> str:
    """Return the model's answer to a single question."""
    resp = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=512,
        messages=[{"role": "user", "content": question}],
    )
    return resp.content[0].text
