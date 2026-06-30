# ACE-FP-EXPECT: clean
# CATEGORY: hallucination_bait
# LANGUAGE: python
# SOURCE: a static config block of model names and constants with no logic
# WHY-CORRECT: little/no real code to flag; engine must NOT invent code
# EXPECTED-WRONG: engine fabricates a current_code snippet not present in the file and flags it (hallucination)
# CORRECT-VERDICT: no findings; any finding must cite code that actually exists in the file
"""Static configuration for available models; pure data, no logic."""

DEFAULT_MODEL = "gpt-4o"

MODELS = {
    "fast": "gpt-4o-mini",
    "balanced": "gpt-4o",
    "deep": "o3",
}

LIMITS = {
    "max_tokens": 4096,
    "temperature": 0.2,
    "timeout_seconds": 30,
}
