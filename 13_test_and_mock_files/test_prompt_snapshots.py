# ACE-FP-EXPECT: clean
# CATEGORY: 13_test_and_mock_files
# SOURCE: Snapshot tests asserting that rendered prompt strings stay stable
# WHY-CORRECT: Hardcoded expected prompt strings are the snapshot baseline; their literalness is intentional so prompt drift is caught in review
# EXPECTED-WRONG: Engine flags the long hardcoded prompt literals as "magic strings" or suggests extracting/parameterizing them, defeating the snapshot's purpose
# CORRECT-VERDICT: no findings
"""Snapshot tests pinning the exact text of rendered prompts."""

from myapp.prompts import render_summary_prompt, render_classify_prompt


def test_summary_prompt_snapshot():
    rendered = render_summary_prompt(doc="The quick brown fox.")
    expected = (
        "You are a precise summarizer.\n"
        "Summarize the document in one sentence.\n\n"
        "Document:\n"
        "The quick brown fox."
    )
    assert rendered == expected


def test_classify_prompt_snapshot():
    rendered = render_classify_prompt(text="my card was declined", labels=["billing", "other"])
    expected = (
        "Classify the text into exactly one label.\n"
        "Labels: billing, other\n\n"
        "Text: my card was declined"
    )
    assert rendered == expected
