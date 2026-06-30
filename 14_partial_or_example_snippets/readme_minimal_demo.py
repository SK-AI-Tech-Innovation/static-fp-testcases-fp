# ACE-FP-EXPECT: clean
# CATEGORY: 14_partial_or_example_snippets
# SOURCE: A README-style minimal demo that prints a single completion
# WHY-CORRECT: A throwaway __main__ demo meant to show the library works in three lines; production concerns (config, logging, error policy) are deliberately out of scope
# EXPECTED-WRONG: Engine demands logging instead of print, error handling, config injection, or "don't hardcode the prompt" — over-spec for a README demo
# CORRECT-VERDICT: no findings
"""Minimal runnable demo for the README."""

from anthropic import Anthropic


if __name__ == "__main__":
    client = Anthropic()
    resp = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=64,
        messages=[{"role": "user", "content": "Say hello in one short sentence."}],
    )
    print(resp.content[0].text)
