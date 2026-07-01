# ACE-FP-EXPECT: clean
# CATEGORY: 08_framework_idioms
# LANGUAGE: python
# SOURCE: ai-readable-data PR #89 (graph.py) — ACE flagged "missing LLM error handling"; author confirmed FP
# WHY-CORRECT: this file only WIRES the graph — it binds node callables via functools.partial and
#              make_*_node factories, then registers nodes/edges on a StateGraph. There is NO LLM call
#              in this file. The actual model calls (and their try/except, retry, failure-state handling)
#              live inside the node functions in other modules. Demanding error handling at the wiring
#              site is misplaced: there is nothing to wrap here.
# EXPECTED-WRONG: engine sees node names like render_ontology / normalize / consolidate and the word
#                 "node", assumes LLM API calls happen here, and flags "missing try/except / retry around
#                 LLM call" — even though this module never invokes a model.
# CORRECT-VERDICT: no findings
"""Graph assembly module: bind node callables and register them on a StateGraph.

Pure wiring. Each node's LLM call and its error handling live in the node's own module;
this file just composes them into a graph.
"""
from functools import partial
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

# Node implementations imported from their own modules (where their LLM calls and
# try/except live). They are referenced here only to be registered.
from .nodes import (
    bootstrap_ontology,
    consolidate_ontology,
    make_render_facts_node,
    normalize_ontology_updates,
    render_ontology_update,
)


class OntologyState(TypedDict):
    documents: list[str]
    ontology: dict
    facts: list[dict]


def build_graph(tools: object) -> object:
    # Bind each node callable to the shared tools — no model is invoked here.
    bootstrap_ontology_node = partial(bootstrap_ontology, tools=tools)  # Phase 4
    render_ontology_node = partial(render_ontology_update, tools=tools)  # Phase 5
    normalize_ontology_node = partial(normalize_ontology_updates, tools=tools)  # Phase 6
    consolidate_ontology_node = partial(consolidate_ontology, tools=tools)  # Phase 7
    render_facts_node = make_render_facts_node(tools)

    graph = StateGraph(OntologyState)
    graph.add_node("bootstrap", bootstrap_ontology_node)
    graph.add_node("render_ontology", render_ontology_node)
    graph.add_node("normalize", normalize_ontology_node)
    graph.add_node("consolidate", consolidate_ontology_node)
    graph.add_node("render_facts", render_facts_node)

    graph.add_edge(START, "bootstrap")
    graph.add_edge("bootstrap", "render_ontology")
    graph.add_edge("render_ontology", "normalize")
    graph.add_edge("normalize", "consolidate")
    graph.add_edge("consolidate", "render_facts")
    graph.add_edge("render_facts", END)
    return graph.compile()
