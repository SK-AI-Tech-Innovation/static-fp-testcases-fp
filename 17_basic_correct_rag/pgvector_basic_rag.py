# ACE-FP-EXPECT: clean
# CATEGORY: 17_basic_correct_rag
# SOURCE: pgvector + psycopg (`psycopg`, `pgvector.psycopg`) + OpenAI
# WHY-CORRECT: complete pgvector RAG — register the vector type, store embeddings in a vector
#              column, retrieve by cosine distance (`<=>`) with a parameterized query, inject
#              the rows into the prompt, answer. SQL is parameterized; ORDER BY uses the cosine op.
# EXPECTED-WRONG: engine invents "add an ivfflat/hnsw index", "add reranking", or "cache results"
# CORRECT-VERDICT: no findings
"""Retrieve documents from a pgvector table by cosine distance and answer the question."""
import psycopg
from openai import OpenAI
from pgvector.psycopg import register_vector

llm = OpenAI()
conn = psycopg.connect("postgresql:///rag")
register_vector(conn)


def embed(text: str) -> list[float]:
    return llm.embeddings.create(model="text-embedding-3-small", input=text).data[0].embedding


def index(documents: list[str]) -> None:
    with conn.cursor() as cur:
        for doc in documents:
            cur.execute(
                "INSERT INTO docs (content, embedding) VALUES (%s, %s)",
                (doc, embed(doc)),
            )
    conn.commit()


def answer(question: str) -> str:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT content FROM docs ORDER BY embedding <=> %s LIMIT 3",
            (embed(question),),
        )
        context = "\n\n".join(row[0] for row in cur.fetchall())
    prompt = f"Answer using the context.\n\nContext:\n{context}\n\nQuestion: {question}"
    response = llm.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    index(["Photosynthesis converts light into chemical energy."])
    print(answer("What does photosynthesis do?"))
