# ACE-FP-EXPECT: clean
# CATEGORY: 06_already_satisfied_observability
# SOURCE: structlog structured logging of model, tokens, cost, and latency per call
# WHY-CORRECT: each completion emits a structured log event with model, token counts, derived
#              USD cost, and measured latency — fully observable, machine-parseable telemetry
# EXPECTED-WRONG: engine flags "no metrics/observability on call" because it expects an APM SDK
#                 or tracer and does not credit a structured `logger.info(..., **fields)` event
# CORRECT-VERDICT: no findings
"""Emit a structured log event with model, tokens, cost, and latency for every call."""
import time

import structlog
from openai import OpenAI

log = structlog.get_logger("llm")
client = OpenAI()

# USD per 1K tokens (illustrative pricing table for cost derivation).
_PRICE_PER_1K = {"gpt-4.1-mini": (0.0004, 0.0016)}


def complete(prompt: str, model: str = "gpt-4.1-mini") -> str:
    start = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    latency_ms = (time.perf_counter() - start) * 1000

    in_rate, out_rate = _PRICE_PER_1K.get(model, (0.0, 0.0))
    usage = resp.usage
    cost_usd = (usage.prompt_tokens * in_rate + usage.completion_tokens * out_rate) / 1000

    log.info(
        "llm_call",
        model=model,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        total_tokens=usage.total_tokens,
        cost_usd=round(cost_usd, 6),
        latency_ms=round(latency_ms, 1),
    )
    return resp.choices[0].message.content
