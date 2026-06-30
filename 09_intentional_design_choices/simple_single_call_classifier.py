# ACE-FP-EXPECT: clean
# CATEGORY: 09_intentional_design_choices
# SOURCE: A tiny one-shot intent classifier for a small fixed label set
# WHY-CORRECT: A single stateless LLM call is the right tool; introducing an agent loop, graph, tool-calling, or orchestration framework would be over-engineering for a one-shot classification
# EXPECTED-WRONG: Engine suggests adding an agent/state machine/multi-step reasoning or "scalable orchestration", which is unnecessary complexity here
# CORRECT-VERDICT: no findings
"""Classify a support message into one of three fixed intents."""

from anthropic import Anthropic

client = Anthropic()

LABELS = ("billing", "technical", "other")


def classify_intent(message: str) -> str:
    """Map a support message to one of LABELS with a single LLM call.

    Args:
        message: The customer's message.

    Returns:
        str: One of "billing", "technical", or "other".
    """
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=8,
        temperature=0,
        system=(
            "Classify the message into exactly one label: "
            "billing, technical, or other. Reply with only the label."
        ),
        messages=[{"role": "user", "content": message}],
    )
    label = response.content[0].text.strip().lower()
    return label if label in LABELS else "other"
