# ACE-FP-EXPECT: clean
# CATEGORY: 14_partial_or_example_snippets
# SOURCE: A module skeleton with stubbed functions awaiting implementation
# WHY-CORRECT: An interface-first skeleton; pass/... bodies and NotImplementedError are intentional placeholders, not bugs to "complete"
# EXPECTED-WRONG: Engine flags empty bodies, the bare ..., or NotImplementedError as missing logic / unhandled paths
# CORRECT-VERDICT: no findings
"""Skeleton for the upcoming retrieval module (implementation pending)."""

from typing import Protocol


class Retriever(Protocol):
    """Interface every retriever backend will implement."""

    def search(self, query: str, k: int = 5) -> list[str]:
        ...


def build_index(documents: list[str]) -> None:
    """Build the vector index from documents. (To be implemented.)"""
    ...


def query_index(query: str, k: int = 5) -> list[str]:
    """Return the top-k passages for a query. (To be implemented.)"""
    raise NotImplementedError


def teardown() -> None:
    """Release index resources. (To be implemented.)"""
    pass
