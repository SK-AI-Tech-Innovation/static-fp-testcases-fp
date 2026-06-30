# ACE-FP-EXPECT: clean
# CATEGORY: 40_advanced_rag
# SOURCE: Microsoft GraphRAG — local search (entity-anchored graph + text-unit retrieval)
# WHY-CORRECT: GraphRAG local search answers entity-specific questions by first finding entities
#   relevant to the query (via embedding similarity over entity descriptions), then EXPANDING the
#   knowledge graph around them — pulling connected entities, relationships, and the text units
#   that mention them — to build a context window. Mixing graph traversal with the text-unit
#   lookup is the intended local-search design.
# EXPECTED-WRONG: engine may flag the graph-expansion step as "retrieving unrelated docs", claim
#   neighbor entities dilute relevance, or that only the top-k entity match should be used.
# CORRECT-VERDICT: no findings
"""GraphRAG local search: anchor on entities, expand the graph, gather connected text units."""
from dataclasses import dataclass, field


@dataclass
class Entity:
    name: str
    description: str
    text_unit_ids: list[str] = field(default_factory=list)
    neighbors: list[str] = field(default_factory=list)


def find_seed_entities(query: str, entities: dict[str, Entity], k: int = 2) -> list[Entity]:
    scored = sorted(
        entities.values(),
        key=lambda e: len(set(query.lower().split()) & set(e.description.lower().split())),
        reverse=True,
    )
    return scored[:k]


def build_local_context(query: str, entities: dict[str, Entity], text_units: dict[str, str]) -> str:
    seeds = find_seed_entities(query, entities)
    selected = {e.name: e for e in seeds}
    for seed in seeds:
        for n in seed.neighbors:
            if n in entities:
                selected[n] = entities[n]
    unit_ids = {uid for e in selected.values() for uid in e.text_unit_ids}
    return "\n".join(text_units[u] for u in unit_ids if u in text_units)


if __name__ == "__main__":
    ents = {
        "Acme": Entity("Acme", "cloud vendor revenue", ["t1"], neighbors=["Beta"]),
        "Beta": Entity("Beta", "supplier partner", ["t2"]),
    }
    units = {"t1": "Acme revenue grew.", "t2": "Beta supplies Acme."}
    print(build_local_context("Acme revenue", ents, units))
