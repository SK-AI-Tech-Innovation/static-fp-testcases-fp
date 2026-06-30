# ACE-FP-EXPECT: clean
# CATEGORY: 40_advanced_rag
# SOURCE: Self-RAG — reflective retrieve/generate/critique loop on a LangGraph StateGraph
# WHY-CORRECT: Self-RAG reflects on its own output: generate -> grade for relevance/groundedness
#   -> if unsupported, rewrite the query and retrieve again, looping until grounded or a max
#   iteration cap is hit. The cycle in the StateGraph is intentional and bounded by an iteration
#   counter, so it terminates. The reflection nodes are the whole point of Self-RAG.
# EXPECTED-WRONG: engine may flag the retrieve<->generate cycle as an infinite loop, claim the
#   self-grading step is redundant, or that re-retrieving on a rewritten query is wasteful.
# CORRECT-VERDICT: no findings
"""Self-RAG: a bounded reflective loop that re-retrieves until the answer is grounded."""
from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class RagState(TypedDict):
    query: str
    docs: list[str]
    answer: str
    grounded: bool
    iterations: int


def retrieve(state: RagState) -> dict:
    return {"docs": [f"doc about {state['query']}"]}


def generate(state: RagState) -> dict:
    return {"answer": f"answer from {state['docs']}", "iterations": state["iterations"] + 1}


def grade(state: RagState) -> dict:
    return {"grounded": state["iterations"] >= 2}


def route(state: RagState) -> str:
    if state["grounded"] or state["iterations"] >= 3:
        return END
    return "retrieve"


def build_graph():
    builder = StateGraph(RagState)
    builder.add_node("retrieve", retrieve)
    builder.add_node("generate", generate)
    builder.add_node("grade", grade)
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", "grade")
    builder.add_conditional_edges("grade", route, ["retrieve", END])
    return builder.compile()


if __name__ == "__main__":
    graph = build_graph()
    out = graph.invoke({"query": "x", "docs": [], "answer": "", "grounded": False, "iterations": 0})
    print(out["answer"], out["grounded"])
