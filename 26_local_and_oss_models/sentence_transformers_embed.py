# ACE-FP-EXPECT: clean
# CATEGORY: 26_local_and_oss_models
# SOURCE: sentence-transformers local embedding model
# WHY-CORRECT: SentenceTransformer loads all-MiniLM-L6-v2 from the local cache and encodes texts
#              to vectors entirely on-device. normalize_embeddings + cosine similarity are standard.
#              There is no embeddings API, key, or remote endpoint.
# EXPECTED-WRONG: engine either misses that this is embedding/AI usage (no openai.embeddings.create)
#                 or recommends hosted-embedding-API concerns like batching against rate limits,
#                 retries, or a remote model name — irrelevant for a local encode call.
# CORRECT-VERDICT: no findings
"""Compute sentence embeddings locally with sentence-transformers."""
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

model = SentenceTransformer("all-MiniLM-L6-v2")


def embed(texts: list[str]):
    return model.encode(texts, normalize_embeddings=True, convert_to_tensor=True)


def most_similar(query: str, candidates: list[str]) -> str:
    query_vec = embed([query])
    candidate_vecs = embed(candidates)
    scores = cos_sim(query_vec, candidate_vecs)[0]
    best_idx = int(scores.argmax())
    return candidates[best_idx]


if __name__ == "__main__":
    docs = ["A cat sat on the mat.", "The stock market fell today.", "Dogs are loyal pets."]
    print(most_similar("information about pets", docs))
