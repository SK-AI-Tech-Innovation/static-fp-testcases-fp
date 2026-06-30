# ACE-FP-EXPECT: clean
# CATEGORY: 39_modern_langgraph_v1
# SOURCE: LangGraph 1.0 — StateGraph + TypedDict state + add_conditional_edges
# WHY-CORRECT: A cyclic graph is the intended LangGraph design. The node loops back to itself
#   via add_conditional_edges until a counter condition routes to END. This is a bounded,
#   deliberate cycle — the canonical LangGraph control-flow pattern — not an accidental
#   infinite loop. State is a TypedDict and reducers/return-dicts merge updates.
# EXPECTED-WRONG: engine may flag "graph has a cycle -> infinite loop" or claim the edge from a
#   node back to an earlier node is a bug, or that StateGraph should be a DAG.
# CORRECT-VERDICT: no findings
"""A deliberately cyclic LangGraph StateGraph that loops a node until a counter hits a limit."""
from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class CountState(TypedDict):
    count: int
    log: list[str]


def increment(state: CountState) -> dict:
    new_count = state["count"] + 1
    return {"count": new_count, "log": state["log"] + [f"tick {new_count}"]}


def should_continue(state: CountState) -> str:
    return "increment" if state["count"] < 3 else END


def build_graph():
    builder = StateGraph(CountState)
    builder.add_node("increment", increment)
    builder.add_edge(START, "increment")
    builder.add_conditional_edges("increment", should_continue, ["increment", END])
    return builder.compile()


if __name__ == "__main__":
    graph = build_graph()
    final = graph.invoke({"count": 0, "log": []})
    print(final["log"])
