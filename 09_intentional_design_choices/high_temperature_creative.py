# ACE-FP-EXPECT: clean
# CATEGORY: 09_intentional_design_choices
# SOURCE: A brainstorming/ideation feature that deliberately wants diverse, non-repetitive output
# WHY-CORRECT: temperature=0.9 is a deliberate choice to maximize variety for a creative naming task; high temperature is exactly right here, not a misconfiguration
# EXPECTED-WRONG: Engine flags temperature>=0.8 as "too high / non-deterministic / unsafe" and suggests lowering it toward 0, ignoring that diversity is the goal
# CORRECT-VERDICT: no findings
"""Generate diverse, creative product name candidates using a high-temperature LLM call."""

from anthropic import Anthropic

client = Anthropic()


def brainstorm_product_names(description: str, n: int = 8) -> list[str]:
    """Return a list of distinct, creative product name ideas.

    This is a divergent ideation task: we *want* surprising, varied output,
    so a high temperature is the correct setting. Determinism would actively
    hurt the user experience by returning the same few safe names every time.

    Args:
        description: A short description of the product.
        n: How many name candidates to request.

    Returns:
        list[str]: Candidate product names, one per line.
    """
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=512,
        temperature=0.9,  # intentional: maximize creative diversity for brainstorming
        system="You are a creative branding assistant. Output one product name per line, no numbering.",
        messages=[
            {
                "role": "user",
                "content": f"Suggest {n} bold, memorable names for: {description}",
            }
        ],
    )
    text = response.content[0].text.strip()
    return [line.strip() for line in text.splitlines() if line.strip()]
