# ACE-FP-EXPECT: clean
# CATEGORY: 13_test_and_mock_files
# SOURCE: A fixture that intentionally emits malformed JSON to exercise the real parser's error handling
# WHY-CORRECT: The "bad" output is the test's input on purpose; the assertion verifies the production code raises/handles correctly. "Fixing" the malformed payloads would destroy the test.
# EXPECTED-WRONG: Engine flags the malformed/invalid JSON payloads as a defect or suggests making the responses valid, defeating the negative test
# CORRECT-VERDICT: no findings
"""Fixtures of deliberately malformed model outputs for error-path testing."""

import pytest

from mypipeline.parse import parse_model_json, ModelOutputError


# Each entry is intentionally broken in a different way to probe the parser.
MALFORMED_OUTPUTS = [
    '{"name": "missing brace"',          # truncated object
    "{'name': 'single quotes'}",          # invalid JSON quoting
    "Here is your answer: {\"name\": 1}",  # leading prose before JSON
    "",                                    # empty completion
]


@pytest.fixture(params=MALFORMED_OUTPUTS)
def malformed_completion(request):
    """Yield one deliberately malformed completion string per parametrization."""
    return request.param


def test_parser_rejects_malformed_output(malformed_completion):
    # The point of this test: the REAL parser must surface a clean,
    # typed error for every malformed payload above — never crash randomly.
    with pytest.raises(ModelOutputError):
        parse_model_json(malformed_completion)
