# ACE-FP-EXPECT: clean
# CATEGORY: 17_basic_correct_rag
# SOURCE: faiss-cpu (`faiss`) IndexFlatIP + OpenAI embeddings/chat
# WHY-CORRECT: standard FAISS RAG — embed docs to a float32 matrix, L2-normalize so inner product
#              equals cosine, build IndexFlatIP, normalize the query, search top-k, map indices
#              back to source texts, inject into the prompt, answer.
# EXPECTED-WRONG: engine invents "use IVF/HNSW index", "add reranking", or "persist the index"
# CORRECT-VERDICT: no findings
"""Build a FAISS index over document embeddings and answer the nearest-match question."""
import faiss
import numpy as np
from openai import OpenAI

llm = OpenAI()


def embed(texts: list[str]) -> np.ndarray:
    response = llm.embeddings.create(model="text-embedding-3-small", input=texts)
    vectors = np.array([item.embedding for item in response.data], dtype="float32")
    faiss.normalize_L2(vectors)
    return vectors


class FaissRAG:
    def __init__(self, documents: list[str]) -> None:
        self.documents = documents
        vectors = embed(documents)
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

    def answer(self, question: str) -> str:
        _, indices = self.index.search(embed([question]), 3)
        context = "\n\n".join(self.documents[i] for i in indices[0])
        prompt = f"Answer using the context.\n\nContext:\n{context}\n\nQuestion: {question}"
        response = llm.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content


if __name__ == "__main__":
    rag = FaissRAG(["Water boils at 100 C at sea level.", "Mercury is the closest planet to the Sun."])
    print(rag.answer("At what temperature does water boil?"))
