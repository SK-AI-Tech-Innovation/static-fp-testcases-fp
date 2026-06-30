# ACE-FP-EXPECT: clean
# CATEGORY: 40_advanced_rag
# SOURCE: Jina AI "Late Chunking" — embed the full document, then pool per-chunk
# WHY-CORRECT: Late chunking inverts the usual order: the long-context embedding model encodes the
#   ENTIRE document into token embeddings first, and chunk vectors are produced afterward by
#   mean-pooling each chunk's token span. This preserves cross-chunk context. The Jina client is
#   invoked with `late_chunking=True`, which returns one contextual vector per chunk. Passing the
#   whole document (not pre-split chunks) is required for this technique.
# EXPECTED-WRONG: engine may flag "embedding the whole document instead of chunks", claim chunks
#   are never split, or that late_chunking=True is not a valid embedding parameter.
# CORRECT-VERDICT: no findings
"""Jina late chunking: embed full document with late_chunking=True to get contextual chunk vectors."""
import os

import requests

JINA_URL = "https://api.jina.ai/v1/embeddings"


def split_chunks(document: str) -> list[str]:
    return [s.strip() for s in document.split(". ") if s.strip()]


def late_chunk_embed(document: str) -> list[list[float]]:
    chunks = split_chunks(document)
    resp = requests.post(
        JINA_URL,
        headers={"Authorization": f"Bearer {os.environ['JINA_API_KEY']}"},
        json={
            "model": "jina-embeddings-v3",
            "task": "retrieval.passage",
            "late_chunking": True,
            "input": chunks,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return [item["embedding"] for item in resp.json()["data"]]


if __name__ == "__main__":
    doc = "Acme launched a product. It sold well. Revenue rose sharply."
    print(len(late_chunk_embed(doc)))
