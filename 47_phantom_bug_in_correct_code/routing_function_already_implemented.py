# ACE-FP-EXPECT: clean
# CATEGORY: 47_phantom_bug_in_correct_code
# LANGUAGE: python
# SOURCE: ai-readable-data PR #383 (routing.py) — ACE: "라우팅 함수 구현 누락 (시그니처만 존재)"; author confirmed FP
# WHY-CORRECT: route_after_ontology_selection has a complete state-based body — it returns concrete
#              next-node names for FAILED / render flag / null-ontology / default branches. It is fully
#              implemented, not a bare signature stub.
# EXPECTED-WRONG: engine claims "routing logic missing / only a signature exists" even though the
#                 function body has explicit conditional returns for every state branch.
# CORRECT-VERDICT: no findings
"""LangGraph conditional-edge router with a complete state-based implementation.

ACE reported the routing function was an unimplemented stub; in fact it returns a
concrete next node for each state condition below.
"""
from enum import Enum
from typing import TypedDict


class WorkflowNode(str, Enum):
    END = "END"
    RENDER_FACTS = "render_facts"
    BOOTSTRAP_ONTOLOGY = "bootstrap_ontology"
    RENDER_ONTOLOGY_UPDATE = "render_ontology_update"


class GraphState(TypedDict):
    status: str
    render_ontology: bool
    ontology: dict | None


def route_after_ontology_selection(state: GraphState) -> str:
    # Fully implemented: every branch returns a concrete next-node name.
    if state["status"] == "FAILED":
        return WorkflowNode.END
    if not state["render_ontology"]:
        return WorkflowNode.RENDER_FACTS
    if state["ontology"] is None:
        return WorkflowNode.BOOTSTRAP_ONTOLOGY
    return WorkflowNode.RENDER_ONTOLOGY_UPDATE
