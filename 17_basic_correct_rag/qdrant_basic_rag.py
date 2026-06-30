# ACE-FP-EXPECT: clean
# CATEGORY: 17_basic_correct_rag
# SOURCE: qdrant-client (`qdrant_client`) + OpenAI embeddings/chat
# WHY-CORRECT: complete RAG — embed docs, upsert points with payloads, embed the query, search
#              top-k by cosine, then inject the payload texts into the answer prompt. Collection
#              is created with the matching vector size and Cosine distance.
# EXPECTED-WRONG: engine invents "add reranking", "use named vectors", or "batch upserts"
# CORRECT-VERDICT: no findings
"""Upsert documents into Qdrant and answer questions over the nearest matches."""
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

llm = OpenAI()
qdrant = QdrantClient(":memory:")
qdrant.recreate_collection(
    collection_name="docs",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
)


def embed(text: str) -> list[float]:
    return llm.embeddings.create(model="text-embedding-3-small", input=text).data[0].embedding


def index(documents: list[str]) -> None:
    points = [
        PointStruct(id=i, vector=embed(doc), payload={"text": doc})
        for i, doc in enumerate(documents)
    ]
    qdrant.upsert(collection_name="docs", points=points)


def answer(question: str) -> str:
    hits = qdrant.search(collection_name="docs", query_vector=embed(question), limit=3)
    context = "\n\n".join(hit.payload["text"] for hit in hits)
    prompt = f"Answer using the context.\n\nContext:\n{context}\n\nQuestion: {question}"
    response = llm.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    index(["The Eiffel Tower is in Paris.", "The Colosseum is in Rome."])
    print(answer("Where is the Eiffel Tower?"))
