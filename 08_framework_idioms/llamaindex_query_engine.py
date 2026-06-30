# ACE-FP-EXPECT: clean
# CATEGORY: 08_framework_idioms
# SOURCE: LlamaIndex VectorStoreIndex + query engine (idiomatic RAG setup)
# WHY-CORRECT: This is the textbook LlamaIndex RAG pattern: load documents, build a
#              VectorStoreIndex, derive a query engine, and .query(). Retrieval, embedding,
#              prompt assembly, and synthesis are all handled by the framework's defaults;
#              the user code stays declarative. Configuring similarity_top_k and a response
#              mode is correct tuning, not a smell.
# EXPECTED-WRONG: a scanner may flag "embedding model not configured" or "no prompt template"
#                 because the embed model + QA prompt are framework defaults (Settings), not
#                 explicit in this file -> spurious "missing embedding/prompt config" finding.
#                 "embedding" here is a real vector embedding but correctly delegated.
# CORRECT-VERDICT: no findings
"""Idiomatic LlamaIndex RAG: build a VectorStoreIndex and query it. Defaults delegated."""
from __future__ import annotations

from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    VectorStoreIndex,
)


def build_query_engine(docs_dir: str, top_k: int = 4) -> "object":
    """Load a directory of documents and return a configured query engine."""
    documents = SimpleDirectoryReader(docs_dir).load_data()
    index = VectorStoreIndex.from_documents(documents)
    return index.as_query_engine(
        similarity_top_k=top_k,
        response_mode="compact",
    )


def configure_models(chunk_size: int = 512) -> None:
    """Optionally tune global Settings; embed/LLM defaults are framework-managed."""
    Settings.chunk_size = chunk_size
    Settings.chunk_overlap = 64


def ask(docs_dir: str, question: str) -> str:
    """Answer a question over the documents using retrieval-augmented generation."""
    engine = build_query_engine(docs_dir)
    response = engine.query(question)
    return str(response)
