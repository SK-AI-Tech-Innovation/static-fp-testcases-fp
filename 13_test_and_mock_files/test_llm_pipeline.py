# ACE-FP-EXPECT: clean
# CATEGORY: 13_test_and_mock_files
# SOURCE: A pytest module exercising a pipeline with a MockLLM returning canned responses
# WHY-CORRECT: This is test code; hardcoded fake responses and naive "parsing" inside the mock are intentional fixtures, not production logic to harden
# EXPECTED-WRONG: Engine treats the mock as production code and flags hardcoded strings, missing error handling, or "fragile parsing" — all expected in a test double
# CORRECT-VERDICT: no findings
"""Unit tests for the extraction pipeline using a deterministic MockLLM."""

import json

from mypipeline.extract import extract_entities


class MockLLM:
    """A deterministic stand-in for the real client used only in tests."""

    def __init__(self, responses):
        self._responses = list(responses)

    def complete(self, prompt: str) -> str:
        # Tests don't care about the prompt; return the next canned reply.
        return self._responses.pop(0)


def test_extract_entities_happy_path():
    fake = MockLLM([json.dumps({"entities": ["Acme", "Bob"]})])
    result = extract_entities("Acme hired Bob", llm=fake)
    assert result == ["Acme", "Bob"]


def test_extract_entities_empty():
    fake = MockLLM([json.dumps({"entities": []})])
    assert extract_entities("nothing here", llm=fake) == []


def test_extract_entities_dedupes():
    fake = MockLLM([json.dumps({"entities": ["Acme", "Acme"]})])
    assert extract_entities("Acme Acme", llm=fake) == ["Acme"]
