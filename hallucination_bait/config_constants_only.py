# ACE-FP-EXPECT: clean
# CATEGORY: hallucination_bait
# LANGUAGE: python
# SOURCE: synthetic; a pure configuration module — constants and dicts, zero logic
# WHY-CORRECT: there are no functions, no LLM calls, no control flow, no sinks; only model names and parameter literals. Naming a model id or "temperature" is not an anti-pattern by itself
# EXPECTED-WRONG: inferring runtime behavior (missing retries, unsafe calls, prompt injection) from config keys/values that are never executed
# CORRECT-VERDICT: no findings
"""Static configuration for the assistant. Constants only — no logic."""

from __future__ import annotations

# Model selection
DEFAULT_CHAT_MODEL = "claude-opus-4-20250514"
FALLBACK_CHAT_MODEL = "gpt-4o"
EMBEDDING_MODEL = "text-embedding-3-small"

# Generation parameters
GENERATION_DEFAULTS = {
    "max_tokens": 1024,
    "temperature": 0.2,
    "top_p": 0.9,
}

# Retry policy parameters (consumed elsewhere; not applied here)
RETRY_POLICY = {
    "max_retries": 4,
    "base_backoff_seconds": 0.5,
    "max_backoff_seconds": 8.0,
    "retry_on_status": [429, 500, 502, 503, 529],
}

# Retrieval parameters
RETRIEVAL = {
    "top_k": 4,
    "rrf_k": 60,
    "cache_ttl_seconds": 300,
}

# Per-task model overrides
TASK_MODELS = {
    "summarize": DEFAULT_CHAT_MODEL,
    "classify": FALLBACK_CHAT_MODEL,
    "extract": FALLBACK_CHAT_MODEL,
}

# Logging configuration
LOGGING = {
    "level": "INFO",
    "format": "json",
    "include_request_id": True,
}
