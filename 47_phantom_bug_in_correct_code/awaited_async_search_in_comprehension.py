# ACE-FP-EXPECT: clean
# CATEGORY: 47_phantom_bug_in_correct_code
# LANGUAGE: python
# SOURCE: ai-readable-data PR #80 (atomic_tools.py) — ACE flagged a missing `await`; author confirmed FP
# WHY-CORRECT: the async provider call IS awaited on its own line, so `raw_hits` is the already-
#              resolved list (not a coroutine) by the time the list comprehension iterates it.
#              `await x` completes before the next statement runs — iterating `raw_hits` afterward
#              is correct and never touches a coroutine object.
# EXPECTED-WRONG: engine invents a bug — "async call used without await, coroutine passed to the
#                 comprehension, RuntimeError" — by pattern-matching `[... for h in raw_hits]` near an
#                 async method and missing that line N already awaited the result into `raw_hits`.
# CORRECT-VERDICT: no findings
"""Web-search tool that awaits the async provider, then maps the resolved hits.

This reproduces the real PR #80 shape: the await and the comprehension are on adjacent
lines. ACE reported "search() called without await -> coroutine in comprehension", but
the call on the previous line is already awaited, so the result is a plain list.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SearchHit:
    title: str
    url: str
    snippet: str


class AtomicToolBox:
    """Minimal tool box exposing an async web search over a pluggable provider."""

    def __init__(self, search_provider: object) -> None:
        self.search_provider = search_provider

    async def search(self, query: str, limit: int = 5) -> list[SearchHit]:
        # The async provider call is fully awaited here — `raw_hits` is a resolved list.
        raw_hits = await self.search_provider.search(query=query, max_results=limit)
        # Iterating an already-resolved list. No coroutine is ever placed in the comprehension.
        return [SearchHit(title=h.title, url=h.url, snippet=h.snippet) for h in raw_hits]
