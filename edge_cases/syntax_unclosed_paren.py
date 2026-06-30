# ACE-FP-EXPECT: clean
# CATEGORY: edge_cases/syntax_error
# LANGUAGE: python
# SOURCE: A Python file with a syntax error (unclosed parenthesis) — AST parse will fail
# WHY-CORRECT: Robustness case, not a correctness case: the engine must NOT crash on unparseable code; the Python AST deterministic rules should degrade to UNKNOWN, not raise
# EXPECTED-WRONG: The engine raises on ast.parse() and the analysis errors out, or it hallucinates findings about code it could not parse
# CORRECT-VERDICT: no crash; no confident findings on unparseable source
"""Summarizer with a deliberately unclosed call — must not crash the static engine."""

from openai import OpenAI

client = OpenAI()


def summarize(text: str) -> str:
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": f"Summarize:\n{text}"}
    # <-- unclosed: the call's closing parenthesis is missing on purpose
    return resp.choices[0].message.content
