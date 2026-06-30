# ACE-FP-EXPECT: clean
# CATEGORY: hallucination_bait
# LANGUAGE: python
# SOURCE: a single correct trivial constant assignment
# WHY-CORRECT: little/no real code to flag; engine must NOT invent code
# EXPECTED-WRONG: engine fabricates a current_code snippet not present in the file and flags it (hallucination)
# CORRECT-VERDICT: no findings; any finding must cite code that actually exists in the file
"""A single trivial, correct constant."""

MODEL = "gpt-5.5"
