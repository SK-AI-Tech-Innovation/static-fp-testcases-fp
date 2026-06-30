# ACE-FP-EXPECT: clean
# CATEGORY: 08_framework_idioms
# SOURCE: Haystack 2.x Pipeline with components wired via add_component/connect
# WHY-CORRECT: This is idiomatic Haystack 2.x: instantiate a Pipeline, add_component() each
#              node, connect("producer.output", "consumer.input") by named sockets, and run()
#              with a dict keyed by entry-component name. The graph-of-components style and
#              string socket connections are the framework's intended API, not ad-hoc glue.
# EXPECTED-WRONG: the string-based connect("retriever.documents", "prompt_builder.documents")
#                 and "prompt_builder" naming may be misread as fragile string coupling or an
#                 unmanaged/raw "prompt" -> spurious "hardcoded prompt / brittle wiring"
#                 finding, though both are the documented Haystack idiom.
# CORRECT-VERDICT: no findings
"""Idiomatic Haystack 2.x retrieval pipeline wired by named component sockets."""
from __future__ import annotations

from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore

PROMPT_TEMPLATE = """\
Given the context below, answer the question.

Context:
{% for doc in documents %}
{{ doc.content }}
{% endfor %}

Question: {{ question }}
Answer:
"""


def build_rag_pipeline(document_store: InMemoryDocumentStore) -> Pipeline:
    """Construct a retriever -> prompt builder -> generator pipeline."""
    pipeline = Pipeline()
    pipeline.add_component("retriever", InMemoryBM25Retriever(document_store))
    pipeline.add_component("prompt_builder", PromptBuilder(template=PROMPT_TEMPLATE))
    pipeline.add_component("generator", OpenAIGenerator(model="gpt-4o-mini"))

    pipeline.connect("retriever.documents", "prompt_builder.documents")
    pipeline.connect("prompt_builder.prompt", "generator.prompt")
    return pipeline


def ask(document_store: InMemoryDocumentStore, question: str) -> str:
    """Run the pipeline; inputs are keyed by the entry components' names."""
    pipeline = build_rag_pipeline(document_store)
    result = pipeline.run(
        {
            "retriever": {"query": question},
            "prompt_builder": {"question": question},
        }
    )
    return result["generator"]["replies"][0]
