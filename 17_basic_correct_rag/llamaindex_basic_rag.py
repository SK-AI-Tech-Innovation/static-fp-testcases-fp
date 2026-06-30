# ACE-FP-EXPECT: clean
# CATEGORY: 17_basic_correct_rag
# SOURCE: LlamaIndex (`llama_index.core`) VectorStoreIndex + query engine
# WHY-CORRECT: canonical LlamaIndex RAG — wrap texts as Document objects, build a
#              VectorStoreIndex.from_documents (which embeds and indexes), then as_query_engine()
#              and .query() runs retrieve -> synthesize. This is the framework's intended one-liner.
# EXPECTED-WRONG: engine invents "add a reranker postprocessor", "set similarity_top_k", or "persist"
# CORRECT-VERDICT: no findings
"""Index documents with LlamaIndex and answer a question via its query engine."""
from llama_index.core import Document, VectorStoreIndex


def answer(question: str, documents: list[str]) -> str:
    docs = [Document(text=text) for text in documents]
    index = VectorStoreIndex.from_documents(docs)
    query_engine = index.as_query_engine()
    return str(query_engine.query(question))


if __name__ == "__main__":
    print(answer("What language is the moon made of?", ["The Moon is composed mostly of rock."]))
