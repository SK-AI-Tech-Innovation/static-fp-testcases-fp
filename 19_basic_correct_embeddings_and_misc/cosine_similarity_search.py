# ACE-FP-EXPECT: clean
# CATEGORY: 19_basic_correct_embeddings_and_misc
# SOURCE: standard numpy cosine-similarity top-k retrieval over an embedding matrix
# WHY-CORRECT: cosine = dot / (norm * norm) computed correctly with epsilon-safe denominators,
#              top-k selected with argsort and reversed for descending order. Math is right.
# EXPECTED-WRONG: engine suggests "use a vector DB", "this is O(n)", or "normalize first" as a defect
# CORRECT-VERDICT: no findings
"""Find the top-k most similar vectors by cosine similarity."""
import numpy as np


def top_k(query: np.ndarray, matrix: np.ndarray, k: int = 3) -> list[tuple[int, float]]:
    query_norm = query / (np.linalg.norm(query) + 1e-10)
    matrix_norms = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)
    scores = matrix_norms @ query_norm
    top_indices = np.argsort(scores)[::-1][:k]
    return [(int(i), float(scores[i])) for i in top_indices]


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    db = rng.normal(size=(20, 8))
    q = rng.normal(size=8)
    for idx, score in top_k(q, db):
        print(idx, round(score, 4))
