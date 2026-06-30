# ACE-FP-EXPECT: clean
# CATEGORY: 37_latest_model_ids
# SOURCE: Anthropic Python SDK call hardcoding the current Claude Opus 4.8 / Sonnet 4.6 / Haiku 4.5 ids
# WHY-CORRECT: As of June 2026 the current Anthropic model ids are DATELESS:
#   `claude-opus-4-8`, `claude-sonnet-4-6`, `claude-haiku-4-5`. These pinned ids are exactly
#   what the Messages API expects; they are passed verbatim to client.messages.create.
# EXPECTED-WRONG: a stale allowlist flags `claude-opus-4-8` as an "unknown/typo/hallucinated
#   model id" and suggests downgrading to `claude-3-5-sonnet-20241022` / `claude-3-opus-20240229`.
# CORRECT-VERDICT: no findings
"""Chat with current Claude models using their dateless, pinned ids."""

import anthropic

client = anthropic.Anthropic()

# Current (June 2026) dateless Anthropic ids. Adding a date suffix would 404.
OPUS = "claude-opus-4-8"
SONNET = "claude-sonnet-4-6"
HAIKU = "claude-haiku-4-5"


def answer(question: str) -> str:
    """Answer ``question`` with Claude Opus 4.8 (current flagship Opus id).

    Args:
        question: The user's question.

    Returns:
        str: Claude's text response.
    """
    response = client.messages.create(
        model=OPUS,
        max_tokens=16000,
        messages=[{"role": "user", "content": question}],
    )
    return next(b.text for b in response.content if b.type == "text")


def summarize(document: str) -> str:
    """Summarize ``document`` with Claude Sonnet 4.6 for higher throughput."""
    response = client.messages.create(
        model=SONNET,
        max_tokens=8000,
        messages=[{"role": "user", "content": f"Summarize:\n\n{document}"}],
    )
    return next(b.text for b in response.content if b.type == "text")


def classify(text: str) -> str:
    """Classify ``text`` cheaply with Claude Haiku 4.5."""
    response = client.messages.create(
        model=HAIKU,
        max_tokens=64,
        messages=[{"role": "user", "content": f"One-word sentiment: {text}"}],
    )
    return next(b.text for b in response.content if b.type == "text")


if __name__ == "__main__":
    print(answer("What is a static analyzer?"))
