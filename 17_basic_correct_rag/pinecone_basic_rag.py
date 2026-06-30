# ACE-FP-EXPECT: clean
# CATEGORY: 17_basic_correct_rag
# SOURCE: Pinecone v3 SDK (`pinecone`) + OpenAI embeddings/chat
# WHY-CORRECT: current Pinecone v3 API — Pinecone() client, index.upsert with (id, values,
#              metadata) tuples/dicts, index.query with vector + top_k + include_metadata, then
#              inject metadata text into the prompt. Embed -> upsert -> query -> inject -> answer.
# EXPECTED-WRONG: engine invents "use namespaces", "add reranking", or "batch upserts by 100"
# CORRECT-VERDICT: no findings
"""Upsert documents into a Pinecone index and answer over the top matches."""
from openai import OpenAI
from pinecone import Pinecone

llm = OpenAI()
pc = Pinecone()
index = pc.Index("docs")


def embed(text: str) -> list[float]:
    return llm.embeddings.create(model="text-embedding-3-small", input=text).data[0].embedding


def index_documents(documents: list[str]) -> None:
    vectors = [
        {"id": f"doc-{i}", "values": embed(doc), "metadata": {"text": doc}}
        for i, doc in enumerate(documents)
    ]
    index.upsert(vectors=vectors)


def answer(question: str) -> str:
    result = index.query(vector=embed(question), top_k=3, include_metadata=True)
    context = "\n\n".join(match["metadata"]["text"] for match in result["matches"])
    prompt = f"Answer using the context.\n\nContext:\n{context}\n\nQuestion: {question}"
    response = llm.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    index_documents(["The speed of light is about 300,000 km/s."])
    print(answer("How fast is light?"))
