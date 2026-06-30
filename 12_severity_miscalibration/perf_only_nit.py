# ACE-FP-EXPECT: 1
# CATEGORY: 12_severity_miscalibration
# SOURCE: a semantic-search helper that recomputes the same query embedding inside a loop
# WHY-CORRECT: the LLM/embedding usage is functionally correct — it embeds the query and ranks candidates.
#              The only issue is a small inefficiency (the identical query embedding is recomputed on every
#              candidate instead of once). Per the rubric, `performance` is NEVER critical; the most this
#              warrants is a single suggestion/warning to hoist the embedding out of the loop.
# EXPECTED-WRONG: escalating this perf nit to `critical` (or emitting multiple findings for one hoist).
# CORRECT-VERDICT: at most 1 suggestion/warning ("hoist query embedding out of loop"), NEVER critical
"""Rank candidates by similarity to a query embedding — correct, but recomputes the query embedding."""
from __future__ import annotations

import math


def embed(text: str) -> list[float]:
    vec = [0.0] * 32
    for i, ch in enumerate(text.lower()):
        vec[(ord(ch) + i) % 32] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def rank_candidates(query: str, candidates: list[str]) -> list[tuple[str, float]]:
    scored: list[tuple[str, float]] = []
    for candidate in candidates:
        # NIT: embed(query) is identical every iteration and could be hoisted above the loop.
        score = cosine(embed(query), embed(candidate))
        scored.append((candidate, score))
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored


if __name__ == "__main__":
    print(rank_candidates("fast database", ["postgres tuning", "cooking recipes", "query latency"]))
