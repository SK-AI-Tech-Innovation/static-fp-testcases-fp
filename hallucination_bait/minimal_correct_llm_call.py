# ACE-FP-EXPECT: clean
# CATEGORY: hallucination_bait
# LANGUAGE: python
# SOURCE: a short, correct LLM call that tempts the engine to invent extra issues/code
# WHY-CORRECT: little/no real code to flag; engine must NOT invent code
# EXPECTED-WRONG: engine fabricates a current_code snippet not present in the file and flags it (hallucination)
# CORRECT-VERDICT: no findings; any finding must cite code that actually exists in the file
"""A minimal, correct chat completion call."""

from openai import OpenAI

client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}],
)
