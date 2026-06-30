# ACE-FP-EXPECT: clean
# CATEGORY: 17_basic_correct_rag
# SOURCE: OpenAI Python SDK (`openai`) embeddings + numpy cosine similarity
# WHY-CORRECT: minimal, correct in-memory RAG — embed docs once, embed the query, rank by cosine
#              similarity over normalized vectors, take the top-k texts, inject into the prompt,
#              answer. No external store needed for a small corpus; the math is correct.
# EXPECTED-WRONG: engine invents "use a vector DB", "add reranking", or "cache embeddings"
# CORRECT-VERDICT: no findings
"""Embed a small corpus, retrieve by cosine similarity, and answer the question."""
import numpy as np
from openai import OpenAI

llm = OpenAI()


def embed(texts: list[str]) -> np.ndarray:
    response = llm.embeddings.create(model="text-embedding-3-small", input=texts)
    vectors = np.array([item.embedding for item in response.data])
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


class EmbeddingRAG:
    def __init__(self, documents: list[str]) -> None:
        self.documents = documents
        self.matrix = embed(documents)

    def answer(self, question: str, k: int = 3) -> str:
        scores = self.matrix @ embed([question])[0]
        top = np.argsort(scores)[::-1][:k]
        context = "\n\n".join(self.documents[i] for i in top)
        prompt = f"Answer using the context.\n\nContext:\n{context}\n\nQuestion: {question}"
        response = llm.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content


if __name__ == "__main__":
    rag = EmbeddingRAG(["Honey never spoils.", "Octopuses have three hearts."])
    print(rag.answer("How many hearts does an octopus have?"))
