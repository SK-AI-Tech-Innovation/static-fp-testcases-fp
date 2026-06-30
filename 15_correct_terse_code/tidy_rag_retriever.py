# ACE-FP-EXPECT: clean
# CATEGORY: 15_correct_terse_code
# SOURCE: A small, clean RAG flow: chunk -> embed -> search -> inject into prompt
# WHY-CORRECT: Each RAG stage is present and ordered correctly; the code is concise but complete and idiomatic. No missing step to flag.
# EXPECTED-WRONG: Engine suggests reranking, caching, hybrid search, chunk-overlap tuning, or "more robust" abstractions — gratuitous additions to already-correct code
# CORRECT-VERDICT: no findings
"""A compact, correct retrieval-augmented generation helper."""

from anthropic import Anthropic

client = Anthropic()


def _chunk(text: str, size: int = 500) -> list[str]:
    """Split text into fixed-size character chunks."""
    return [text[i : i + size] for i in range(0, len(text), size)]


def answer(question: str, documents: list[str], embed, index) -> str:
    """Answer a question grounded in the most relevant document chunks.

    Args:
        question: The user's question.
        documents: Source documents to ground the answer in.
        embed: Callable mapping text -> vector.
        index: Vector store with add(vectors, payloads) and search(vector, k).

    Returns:
        str: The grounded answer.
    """
    chunks = [c for doc in documents for c in _chunk(doc)]
    index.add([embed(c) for c in chunks], chunks)

    hits = index.search(embed(question), k=4)
    context = "\n\n".join(hits)

    resp = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=600,
        system="Answer using only the provided context. If unknown, say so.",
        messages=[
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ],
    )
    return resp.content[0].text
