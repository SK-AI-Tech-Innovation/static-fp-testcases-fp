# ACE-FP-EXPECT: clean
# CATEGORY: 19_basic_correct_embeddings_and_misc
# SOURCE: sentence-transformers CrossEncoder reranking of retrieved candidates
# WHY-CORRECT: builds (query, candidate) pairs, scores them with a cross-encoder, and returns the
#              candidates sorted by descending relevance. This is the standard two-stage rerank.
# EXPECTED-WRONG: engine suggests "add score threshold" or "normalize scores" as if missing — neither
#                 is required for a correct top-k rerank.
# CORRECT-VERDICT: no findings
"""Rerank retrieved candidates with a cross-encoder."""
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank(query: str, candidates: list[str], top_n: int = 3) -> list[str]:
    pairs = [(query, candidate) for candidate in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
    return [candidate for candidate, _ in ranked[:top_n]]


if __name__ == "__main__":
    docs = [
        "Cats are small domesticated carnivores.",
        "The capital of France is Paris.",
        "Photosynthesis converts light into chemical energy.",
    ]
    print(rerank("What is the capital of France?", docs))
