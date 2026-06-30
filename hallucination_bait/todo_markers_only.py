# ACE-FP-EXPECT: clean
# CATEGORY: hallucination_bait
# LANGUAGE: python
# SOURCE: a module with only TODO markers and `pass`
# WHY-CORRECT: little/no real code to flag; engine must NOT invent code
# EXPECTED-WRONG: engine fabricates a current_code snippet not present in the file and flags it (hallucination)
# CORRECT-VERDICT: no findings; any finding must cite code that actually exists in the file
"""Placeholder module tracking future LLM integration work via TODOs."""


def build_prompt():
    # TODO: assemble the system and user messages
    # TODO: inject retrieved context chunks
    pass


def call_model():
    # TODO: wire up the chat completions client
    # TODO: add retry and timeout handling
    pass
