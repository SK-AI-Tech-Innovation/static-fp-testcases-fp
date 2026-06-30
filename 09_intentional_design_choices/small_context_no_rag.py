# ACE-FP-EXPECT: clean
# CATEGORY: 09_intentional_design_choices
# SOURCE: A Q&A helper that answers from a small, fixed reference text passed inline
# WHY-CORRECT: When the source material is tiny and fixed (a short policy snippet), passing it inline in the prompt is correct; adding a vector store / retrieval layer would be unnecessary infrastructure for a few hundred tokens
# EXPECTED-WRONG: Engine flags "no retrieval / RAG for grounding" or "context should come from a vector DB" and recommends embeddings+retrieval, ignoring that the whole context already fits in the prompt
# CORRECT-VERDICT: no findings
"""Answer questions about a small, fixed policy snippet inline, without any RAG pipeline."""

from anthropic import Anthropic

client = Anthropic()

# The entire reference corpus is this short, stable snippet. It fits comfortably
# in-context, so retrieval/embeddings would be unnecessary infrastructure.
REFUND_POLICY = """\
Refund Policy:
- Digital goods are refundable within 14 days of purchase if unused.
- Subscriptions can be cancelled any time; the current period is not prorated.
- Refunds are issued to the original payment method within 5 business days.
"""


def answer_policy_question(question: str) -> str:
    """Answer a refund-policy question grounded in the inline policy text.

    The full source fits in the prompt, so the answer is grounded by passing
    REFUND_POLICY directly. No RAG / vector store is needed or warranted.

    Args:
        question: A user's question about the refund policy.

    Returns:
        str: An answer grounded only in the provided policy.
    """
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=300,
        temperature=0,
        system=(
            "Answer using only the policy below. If the answer is not covered, "
            f"say you don't know.\n\n{REFUND_POLICY}"
        ),
        messages=[{"role": "user", "content": question}],
    )
    return response.content[0].text.strip()
