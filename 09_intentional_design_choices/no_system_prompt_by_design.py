# ACE-FP-EXPECT: clean
# CATEGORY: 09_intentional_design_choices
# SOURCE: A pure text-transformation utility where the instruction lives entirely in the user turn
# WHY-CORRECT: For a single deterministic transformation, putting the full instruction in the user message and omitting a system prompt is a valid, idiomatic choice; a system prompt would be redundant ceremony
# EXPECTED-WRONG: Engine flags "missing system prompt" as a best-practice violation and recommends always adding a system role, ignoring that it adds no value for this self-contained transformation
# CORRECT-VERDICT: no findings
"""Convert a snippet of Markdown to plain text via a single self-contained LLM instruction."""

from anthropic import Anthropic

client = Anthropic()


def markdown_to_plaintext(markdown: str) -> str:
    """Strip Markdown formatting and return readable plain text.

    The task is fully specified in the user turn, so there is intentionally
    no system prompt. Adding one would only duplicate the instruction without
    changing behavior. temperature=0 keeps the transformation stable.

    Args:
        markdown: A Markdown-formatted string.

    Returns:
        str: The plain-text rendering.
    """
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        temperature=0,
        # No system prompt by design: the instruction is self-contained below.
        messages=[
            {
                "role": "user",
                "content": (
                    "Remove all Markdown syntax and return only the readable "
                    f"plain text. Output nothing else.\n\n{markdown}"
                ),
            }
        ],
    )
    return response.content[0].text.strip()
