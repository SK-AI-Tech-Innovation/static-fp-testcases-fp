# ACE-FP-EXPECT: clean
# CATEGORY: 35_reasoning_model_constraints
# SOURCE: claude-opus-4-8 + anthropic-python (Messages API)
# WHY-CORRECT: On Claude Opus 4.7/4.8 (and Fable 5) the temperature and top_p params were removed; sending them returns a 400. This call correctly omits them and relies on the model's default behavior.
# EXPECTED-WRONG: stale analyzer flags "missing temperature, set temperature=0 for reproducibility"; adding temperature to an Opus 4.8 request 400s.
# CORRECT-VERDICT: no findings
"""claude-opus-4-8 Messages call with no temperature/top_p (those params were removed and now 400)."""

import anthropic

client = anthropic.Anthropic()


def draft_email(topic: str) -> str:
    """Draft an email with Opus 4.8; sampling params are intentionally absent."""
    message = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=1024,
        system="You write clear, professional emails.",
        messages=[{"role": "user", "content": f"Draft a short email about: {topic}"}],
    )
    return message.content[0].text


if __name__ == "__main__":
    print(draft_email("rescheduling tomorrow's standup to 10am"))
