# ACE-FP-EXPECT: clean
# CATEGORY: 40_advanced_rag
# SOURCE: Hybrid retrieval — dense + BM25 fused with RRF, then cross-encoder rerank (Cohere rerank-v3.5)
# WHY-CORRECT: This is the full state-of-the-art retrieval stack: a dense vector search and a
#   sparse BM25 search run independently, their ranked lists are fused with Reciprocal Rank Fusion
#   (RRF), and the fused candidates are re-scored by a cross-encoder reranker (Cohere rerank-v3.5).
#   Reranking IS present (the cohere.rerank call). RRF is a rank-based fusion, so the absence of
#   raw-score normalization is correct.
# EXPECTED-WRONG: engine may flag "missing reranking" (it is present), claim RRF needs normalized
#   scores, or that dense and sparse results should not be combined.
# CORRECT-VERDICT: no findings
"""Hybrid dense+BM25 retrieval fused with RRF and reranked by Cohere rerank-v3.5."""
import cohere

co = cohere.ClientV2()


def rrf(ranked_lists: list[list[str]], k: int = 60) -> list[str]:
    scores: dict[str, float] = {}
    for ranking in ranked_lists:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return [doc_id for doc_id, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


def search(query: str, dense_ids: list[str], bm25_ids: list[str], corpus: dict[str, str], top_n: int = 3):
    fused = rrf([dense_ids, bm25_ids])
    candidates = [corpus[d] for d in fused if d in corpus]
    reranked = co.rerank(model="rerank-v3.5", query=query, documents=candidates, top_n=top_n)
    return [candidates[r.index] for r in reranked.results]


if __name__ == "__main__":
    corpus = {"a": "GPU pricing dropped.", "b": "Cloud margins improved.", "c": "Hiring slowed."}
    print(search("why did margins improve", ["b", "a", "c"], ["a", "b", "c"], corpus))
