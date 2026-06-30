# ACE-FP-EXPECT: clean
# CATEGORY: 39_modern_langgraph_v1
# SOURCE: LangGraph 1.0 — BaseStore for cross-thread long-term memory
# WHY-CORRECT: LangGraph separates short-term memory (checkpointer, per-thread) from long-term
#   memory (a Store, shared across threads). InMemoryStore with namespaced put/get/search is the
#   documented long-term-memory API; the store is passed to `.compile(store=...)` and injected
#   into nodes. Namespacing by (user_id, "memories") is the intended scoping pattern.
# EXPECTED-WRONG: engine may flag InMemoryStore/store= as unknown, claim the checkpointer should
#   hold this data, or that put/search on a store is not a valid LangGraph API.
# CORRECT-VERDICT: no findings
"""Store and retrieve cross-thread long-term memory via a LangGraph BaseStore."""
from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore


class MemState(TypedDict):
    user_id: str
    fact: str


def remember(state: MemState, *, store: BaseStore) -> dict:
    namespace = (state["user_id"], "memories")
    store.put(namespace, key="last_fact", value={"text": state["fact"]})
    hits = store.search(namespace, query=state["fact"])
    return {"fact": hits[0].value["text"] if hits else state["fact"]}


def build_graph():
    builder = StateGraph(MemState)
    builder.add_node("remember", remember)
    builder.add_edge(START, "remember")
    builder.add_edge("remember", END)
    return builder.compile(checkpointer=InMemorySaver(), store=InMemoryStore())


if __name__ == "__main__":
    graph = build_graph()
    config = {"configurable": {"thread_id": "t1"}}
    out = graph.invoke({"user_id": "u1", "fact": "likes espresso"}, config=config)
    print(out["fact"])
