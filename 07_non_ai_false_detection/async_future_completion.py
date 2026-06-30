# ACE-FP-EXPECT: clean
# CATEGORY: 07_non_ai_false_detection
# SOURCE: asyncio Future completion + a prefix-trie autocomplete (stdlib only)
# WHY-CORRECT: "completion" here means (1) resolving/completing an asyncio.Future and
#              (2) text autocompletion (word suggestions from a trie). Neither is an LLM
#              chat/text completion; there is no model, no API client, no prompt.
# EXPECTED-WRONG: keyword "completion" (set_completion, complete, Completion) -> false
#                 "LLM completion API" detection -> spurious findings about chat APIs.
# CORRECT-VERDICT: no findings
"""asyncio Future completion and trie-based autocomplete. Not an LLM completion."""
from __future__ import annotations

import asyncio


class CompletionGate:
    """Wraps a Future that other coroutines await until it is completed."""

    def __init__(self) -> None:
        self._future: asyncio.Future[str] = asyncio.get_event_loop().create_future()

    def set_completion(self, value: str) -> None:
        """Complete the underlying Future, unblocking awaiters."""
        if not self._future.done():
            self._future.set_result(value)

    async def wait_for_completion(self) -> str:
        """Await completion of the Future and return its result."""
        return await self._future


class Autocomplete:
    """A prefix trie returning word completions for a typed fragment."""

    def __init__(self, words: list[str]) -> None:
        self._words = sorted(set(words))

    def complete(self, prefix: str, limit: int = 5) -> list[str]:
        """Return up to `limit` words that begin with `prefix`."""
        matches = [w for w in self._words if w.startswith(prefix)]
        return matches[:limit]


async def demo() -> tuple[str, list[str]]:
    gate = CompletionGate()
    loop = asyncio.get_event_loop()
    loop.call_soon(gate.set_completion, "ready")
    result = await gate.wait_for_completion()
    suggestions = Autocomplete(["apple", "apply", "apt", "banana"]).complete("app")
    return result, suggestions
