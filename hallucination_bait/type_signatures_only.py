# ACE-FP-EXPECT: clean
# CATEGORY: hallucination_bait
# LANGUAGE: python
# SOURCE: synthetic; a typing/Protocol stub module with no implementations
# WHY-CORRECT: every function/method body is `...` or `pass`; there is no LLM call, no sink, no control flow — nothing to analyze for runtime behavior
# EXPECTED-WRONG: inventing findings about missing retries, missing validation, or unsafe sinks in methods that have no body at all
# CORRECT-VERDICT: no findings
"""Interface definitions only — signatures and type hints, no logic.

This is the kind of file a `*.pyi`-style stub or a Protocol declaration would
contain. There is nothing to execute and therefore no behavior to flag.
"""

from __future__ import annotations

from typing import Any, Protocol, Sequence


class ChatModel(Protocol):
    """Structural type for a chat completion backend."""

    def complete(self, system: str, messages: Sequence[dict[str, Any]]) -> str:
        ...

    def stream(self, system: str, messages: Sequence[dict[str, Any]]) -> Any:
        ...


class Retriever(Protocol):
    """Structural type for a document retriever."""

    def index(self, documents: Sequence[dict[str, Any]]) -> None:
        ...

    def search(self, query: str, top_k: int = 4) -> list[dict[str, Any]]:
        ...


class Extractor(Protocol):
    """Structural type for a structured-extraction backend."""

    def extract(self, text: str, schema: dict[str, Any]) -> dict[str, Any]:
        ...


def build_pipeline(
    model: ChatModel,
    retriever: Retriever,
    extractor: Extractor,
) -> tuple[ChatModel, Retriever, Extractor]:
    """Signature only; no implementation."""
    ...


def answer(model: ChatModel, retriever: Retriever, question: str) -> str:
    """Signature only; no implementation."""
    pass
