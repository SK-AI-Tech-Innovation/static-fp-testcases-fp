# ACE-FP-EXPECT: clean
# CATEGORY: 09_intentional_design_choices
# SOURCE: A one-off CLI maintenance script that makes a single synchronous LLM call
# WHY-CORRECT: A short, sequential CLI script has no concurrency to exploit; a synchronous call is the simplest correct choice and async would add complexity (event loop, await plumbing) for zero benefit
# EXPECTED-WRONG: Engine flags the synchronous (non-async) client call as "blocking / should use async client / await" and recommends asyncio, which would be over-engineering for a one-shot script
# CORRECT-VERDICT: no findings
"""One-off CLI script that generates a release-note summary with a single synchronous LLM call."""

import sys

from anthropic import Anthropic

client = Anthropic()


def summarize_changelog(changelog: str) -> str:
    """Summarize a raw changelog into a short release note.

    This runs inside a one-shot CLI invocation with no concurrent work, so a
    plain synchronous call is intentionally the right design. Introducing
    asyncio here would add an event loop and await boilerplate for no gain.

    Args:
        changelog: The raw changelog text.

    Returns:
        str: A concise release-note summary.
    """
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=512,
        temperature=0.2,
        system="Summarize the changelog into 3-5 user-facing release-note bullets.",
        messages=[{"role": "user", "content": changelog}],
    )
    return response.content[0].text.strip()


def main() -> None:
    """Read a changelog from a file path argument and print the summary."""
    path = sys.argv[1]
    with open(path, encoding="utf-8") as fh:
        changelog = fh.read()
    print(summarize_changelog(changelog))


if __name__ == "__main__":
    main()
