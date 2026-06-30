# ACE-FP-EXPECT: clean
# CATEGORY: hallucination_bait
# LANGUAGE: python
# SOURCE: synthetic; decorated tool/function stubs that raise NotImplementedError
# WHY-CORRECT: each registered tool stub only raises NotImplementedError; there is no LLM call, no sink, no validation gap to assess — the bodies do nothing but signal "not implemented"
# EXPECTED-WRONG: flagging the decorated stubs for missing schemas, missing retries, or unsafe execution when they contain no real logic at all
# CORRECT-VERDICT: no findings
"""A registry of tool stubs. Every implementation is intentionally absent.

The decorator just records the function name in a registry. Each tool body
raises NotImplementedError — there is nothing to execute and nothing to flag.
"""

from __future__ import annotations

from typing import Any, Callable

_REGISTRY: dict[str, Callable[..., Any]] = {}


def tool(func: Callable[..., Any]) -> Callable[..., Any]:
    """Register a function as a tool by name. No behavior beyond registration."""
    _REGISTRY[func.__name__] = func
    return func


@tool
def search_web(query: str) -> list[dict[str, Any]]:
    """Search the web. Not implemented."""
    raise NotImplementedError("search_web is a stub")


@tool
def read_file(path: str) -> str:
    """Read a file from the workspace. Not implemented."""
    raise NotImplementedError("read_file is a stub")


@tool
def run_query(sql: str) -> list[dict[str, Any]]:
    """Run a database query. Not implemented."""
    raise NotImplementedError("run_query is a stub")


@tool
def send_message(channel: str, body: str) -> None:
    """Send a message to a channel. Not implemented."""
    raise NotImplementedError("send_message is a stub")


def registered_tools() -> list[str]:
    """Return the names of all registered tool stubs."""
    return sorted(_REGISTRY)
