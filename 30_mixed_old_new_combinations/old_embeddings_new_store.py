# ACE-FP-EXPECT: clean
# CATEGORY: 30_mixed_old_new_combinations
# SOURCE: openai-python v1.x embeddings (text-embedding-ada-002, 1536-dim) written to a current qdrant-client store
# WHY-CORRECT: ada-002 is an older but still-served model producing 1536-dim vectors; a Qdrant collection sized to 1536 with COSINE distance accepts them exactly. Old embedding model into a modern vector DB is dimension-consistent and valid.
# EXPECTED-WRONG: engine may claim ada-002 is "deprecated/unavailable" or that an old model can't be used with a current Qdrant client.
# CORRECT-VERDICT: no findings
"""Embed with the legacy ada-002 model and upsert into a modern Qdrant store."""

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

oai = OpenAI()
qdrant = QdrantClient(":memory:")

COLLECTION = "docs"
ADA_DIM = 1536


def ensure_collection() -> None:
    qdrant.recreate_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=ADA_DIM, distance=Distance.COSINE),
    )


def index(doc_id: int, text: str) -> None:
    embedding = oai.embeddings.create(
        model="text-embedding-ada-002",
        input=text,
    ).data[0].embedding

    qdrant.upsert(
        collection_name=COLLECTION,
        points=[PointStruct(id=doc_id, vector=embedding, payload={"text": text})],
    )


if __name__ == "__main__":
    ensure_collection()
    index(1, "The mitochondria is the powerhouse of the cell.")
