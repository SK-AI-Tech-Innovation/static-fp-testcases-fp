# ACE-FP-EXPECT: clean
# CATEGORY: 09_intentional_design_choices
# SOURCE: A versioned, code-reviewed prompt stored as a module-level constant
# WHY-CORRECT: Hardcoding a prompt as a named module constant is a deliberate way to version it in git, review it in PRs, and keep it close to the code that uses it; it is not a magic string to externalize
# EXPECTED-WRONG: Engine flags the inline/hardcoded prompt string as "should be externalized to config / prompt registry" or "magic string", contradicting the intentional in-repo versioning strategy
# CORRECT-VERDICT: no findings
"""Classify support tickets using a deliberately version-controlled prompt constant."""

from anthropic import Anthropic

client = Anthropic()

# Versioned prompt: kept in source so changes go through code review and git history.
# Bump PROMPT_VERSION when editing CLASSIFIER_SYSTEM_PROMPT to keep evals comparable.
PROMPT_VERSION = "v3"
CLASSIFIER_SYSTEM_PROMPT = """\
You are a support-ticket triage classifier.
Read the ticket and respond with exactly one label from this set:
BILLING, TECHNICAL, ACCOUNT, FEEDBACK, OTHER.
Respond with the label only, in uppercase, with no extra text.
"""


def classify_ticket(ticket_text: str) -> str:
    """Return a single triage label for a support ticket.

    Args:
        ticket_text: The raw ticket body.

    Returns:
        str: One of BILLING, TECHNICAL, ACCOUNT, FEEDBACK, OTHER.
    """
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=16,
        temperature=0,
        system=CLASSIFIER_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": ticket_text}],
    )
    return response.content[0].text.strip().upper()
