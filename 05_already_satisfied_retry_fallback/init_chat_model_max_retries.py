# ACE-FP-EXPECT: clean
# CATEGORY: 05_already_satisfied_retry_fallback
# LANGUAGE: python
# SOURCE: ai-readable-data PR #80 (llm.py call_llm) — author: HTTP retry handled by init_chat_model max_retries
# WHY-CORRECT: the chat model is built via LangChain `init_chat_model(..., max_retries=6)`, so transient
#              HTTP failures (429 / 5xx / connection errors) are retried with backoff INSIDE the model
#              client. `call_llm` is a deliberately thin wrapper that delegates resilience to that model
#              config — the retry requirement is satisfied at the client layer, not missing.
# EXPECTED-WRONG: engine flags "LLM call has no retry/fallback" at the `await model.ainvoke(...)` site,
#                 not recognizing that max_retries on the init_chat_model-built client already provides
#                 bounded retry with backoff (the same way openai SDK max_retries counts as handled).
# CORRECT-VERDICT: no findings
"""Thin LLM-call wrapper over a model configured with init_chat_model(max_retries=6).

Transient-error retry is configured on the model itself, so the call site does not need
its own retry loop. This is the same principle as openai_max_retries_builtin.py — relying
on the client's built-in bounded retry is correct and minimal.
"""
from __future__ import annotations

from typing import Any, TypeVar

from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import BasePromptTemplate

T = TypeVar("T")

# Built once; max_retries=6 means the client retries transient HTTP failures with backoff.
_model = init_chat_model("azure_openai:gpt-4o", temperature=0, max_retries=6)


async def call_llm(
    prompt: BasePromptTemplate,
    parser: BaseOutputParser[T],
    prompt_kwargs: dict[str, Any],
) -> T:
    # No hand-rolled retry here on purpose: the model's max_retries=6 already covers
    # transient 429/5xx/connection errors with bounded exponential backoff.
    response = await _model.ainvoke(prompt.format_prompt(**prompt_kwargs))
    return parser.parse(response.content)
