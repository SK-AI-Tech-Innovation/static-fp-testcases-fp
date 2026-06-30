# ACE-FP-EXPECT: clean
# CATEGORY: 08_framework_idioms
# SOURCE: LangGraph StateGraph (TypedDict state, nodes, conditional edges, checkpointer)
# WHY-CORRECT: This is the canonical, idiomatic LangGraph pattern: a TypedDict state schema,
#              plain-function nodes, add_conditional_edges for routing, START/END sentinels,
#              and compile(checkpointer=MemorySaver()) for thread-scoped persistence. The
#              "model" call is a deliberately stubbed pure function so the file is self-
#              contained; the graph wiring is exactly what real code looks like.
# EXPECTED-WRONG: unusual control-flow-as-data (nodes return partial state dicts; routing via
#                 a string-returning function) may be misread as a missing/blocking pattern,
#                 e.g. "no error handling around LLM call" or "agent loop without max-iters",
#                 even though the graph + recursion_limit handle that idiomatically.
# CORRECT-VERDICT: no findings
"""Idiomatic LangGraph StateGraph with conditional routing and a memory checkpointer."""
from __future__ import annotations

from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages


class ReviewState(TypedDict):
    """Shared state threaded through every node of the graph."""

    messages: Annotated[list, add_messages]
    draft: str
    revisions: int
    approved: bool


def write_node(state: ReviewState) -> dict:
    """Produce or revise a draft; returns only the keys it updates."""
    revisions = state["revisions"] + 1
    draft = f"draft v{revisions}: {state['messages'][-1].content if state['messages'] else ''}"
    return {"draft": draft, "revisions": revisions}


def critique_node(state: ReviewState) -> dict:
    """Approve once a draft has been revised at least twice."""
    return {"approved": state["revisions"] >= 2}


def route_after_critique(state: ReviewState) -> str:
    """Conditional edge: loop back to writing or finish."""
    if state["approved"] or state["revisions"] >= 5:
        return "done"
    return "revise"


def build_graph() -> "object":
    """Wire nodes and edges, then compile with a checkpointer for persistence."""
    builder = StateGraph(ReviewState)
    builder.add_node("write", write_node)
    builder.add_node("critique", critique_node)

    builder.add_edge(START, "write")
    builder.add_edge("write", "critique")
    builder.add_conditional_edges(
        "critique",
        route_after_critique,
        {"revise": "write", "done": END},
    )

    return builder.compile(checkpointer=MemorySaver())


def run_once(topic: str) -> ReviewState:
    """Invoke the graph under a stable thread_id so state persists across steps."""
    graph = build_graph()
    config = {"configurable": {"thread_id": "review-1"}}
    initial: ReviewState = {
        "messages": [],
        "draft": topic,
        "revisions": 0,
        "approved": False,
    }
    return graph.invoke(initial, config=config)
