# ACE-FP-EXPECT: clean
# CATEGORY: hallucination_bait
# LANGUAGE: python
# SOURCE: stub function signatures whose bodies are `...` or `raise NotImplementedError`
# WHY-CORRECT: little/no real code to flag; engine must NOT invent code
# EXPECTED-WRONG: engine fabricates a current_code snippet not present in the file and flags it (hallucination)
# CORRECT-VERDICT: no findings; any finding must cite code that actually exists in the file
"""Unimplemented stubs for an LLM client interface."""


def embed_text(text):
    ...


def chat_complete(messages, model):
    ...


def stream_chat(messages, model):
    raise NotImplementedError


def count_tokens(text, model):
    raise NotImplementedError("token counting not wired up yet")
