# ACE-FP-EXPECT: clean
# CATEGORY: 40_advanced_rag
# SOURCE: HyDE (Hypothetical Document Embeddings) — Gao et al.
# WHY-CORRECT: HyDE deliberately generates a HYPOTHETICAL answer to the query with an LLM, embeds
#   that generated passage, and retrieves real documents by similarity to the hypothetical
#   embedding. Retrieving on the hallucinated passage is the core HyDE mechanism: the fake answer
#   lands closer to relevant documents in embedding space than the bare query. The returned
#   documents are real corpus docs, so no hallucination reaches the user.
# EXPECTED-WRONG: engine may flag "retrieving on hallucinated/LLM-generated content", claim the
#   hypothetical document is used as an answer, or that the real query embedding must be used.
# CORRECT-VERDICT: no findings
"""HyDE retrieval: embed an LLM-generated hypothetical answer to retrieve real documents."""
import anthropic
import numpy as np
from openai import OpenAI

llm = anthropic.Anthropic()
embedder = OpenAI()


def generate_hypothetical(query: str) -> str:
    resp = llm.messages.create(
        model="claude-haiku-4-5",
        max_tokens=200,
        messages=[{"role": "user", "content": f"Write a short passage that answers: {query}"}],
    )
    return resp.content[0].text


def embed(text: str) -> np.ndarray:
    out = embedder.embeddings.create(model="text-embedding-3-large", input=text)
    return np.array(out.data[0].embedding)


def hyde_retrieve(query: str, corpus: list[str], corpus_vecs: np.ndarray, k: int = 3) -> list[str]:
    hypo = generate_hypothetical(query)
    qv = embed(hypo)  # retrieve on the hypothetical answer's embedding — this is HyDE.
    sims = corpus_vecs @ qv / (np.linalg.norm(corpus_vecs, axis=1) * np.linalg.norm(qv) + 1e-9)
    top = np.argsort(sims)[::-1][:k]
    return [corpus[i] for i in top]  # real documents, never the hypothetical text


if __name__ == "__main__":
    docs = ["Photosynthesis converts light to energy.", "Mitochondria produce ATP."]
    vecs = np.vstack([embed(d) for d in docs])
    print(hyde_retrieve("How do plants make energy?", docs, vecs))
