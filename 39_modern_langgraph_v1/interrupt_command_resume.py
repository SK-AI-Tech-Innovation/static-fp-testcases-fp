# ACE-FP-EXPECT: clean
# CATEGORY: 39_modern_langgraph_v1
# SOURCE: LangGraph 1.0 — human-in-the-loop via interrupt() + Command(resume=...)
# WHY-CORRECT: The modern HITL primitive is calling `interrupt(payload)` inside a node to pause
#   execution; the run is resumed by re-invoking the graph with `Command(resume=value)`. This
#   requires a checkpointer and a thread_id in config, both present here. The node returning
#   after interrupt() is correct: on resume, interrupt() returns the supplied value.
# EXPECTED-WRONG: engine may flag interrupt() as undefined/blocking, claim Command(resume=...)
#   is not a valid invoke argument, or say the node "never returns" after interrupt().
# CORRECT-VERDICT: no findings
"""Human-in-the-loop approval using interrupt() and resumption via Command(resume=...)."""
from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class ApprovalState(TypedDict):
    amount: int
    approved: bool


def request_approval(state: ApprovalState) -> dict:
    decision = interrupt({"question": "Approve payment?", "amount": state["amount"]})
    return {"approved": decision == "yes"}


def build_graph():
    builder = StateGraph(ApprovalState)
    builder.add_node("request_approval", request_approval)
    builder.add_edge(START, "request_approval")
    builder.add_edge("request_approval", END)
    return builder.compile(checkpointer=InMemorySaver())


if __name__ == "__main__":
    graph = build_graph()
    config = {"configurable": {"thread_id": "pay-1"}}
    graph.invoke({"amount": 100, "approved": False}, config=config)
    final = graph.invoke(Command(resume="yes"), config=config)
    print(final["approved"])
