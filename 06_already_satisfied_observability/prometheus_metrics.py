# ACE-FP-EXPECT: clean
# CATEGORY: 06_already_satisfied_observability
# SOURCE: prometheus_client + openai
# WHY-CORRECT: Token counts are recorded to a Prometheus Counter and request latency to a Histogram on every call, exposed via the metrics endpoint. Usage and performance are observable through Prometheus/Grafana, so logging/metrics is present.
# EXPECTED-WRONG: missing observability / no token usage or latency tracking on the LLM call
# CORRECT-VERDICT: no findings
"""Record LLM token usage and latency to Prometheus counters and histograms."""

import time

from openai import OpenAI
from prometheus_client import Counter, Histogram

client = OpenAI(timeout=30)

LLM_TOKENS = Counter(
    "llm_tokens_total",
    "Total tokens consumed by the LLM.",
    ["model", "token_type"],
)
LLM_LATENCY = Histogram(
    "llm_request_latency_seconds",
    "Latency of LLM requests in seconds.",
    ["model"],
)


def chat(prompt: str, model: str = "gpt-4o-mini") -> str:
    """Emit token-count and latency metrics to Prometheus for each call."""
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    LLM_LATENCY.labels(model=model).observe(time.perf_counter() - start)

    usage = response.usage
    LLM_TOKENS.labels(model=model, token_type="prompt").inc(usage.prompt_tokens)
    LLM_TOKENS.labels(model=model, token_type="completion").inc(usage.completion_tokens)

    return response.choices[0].message.content
