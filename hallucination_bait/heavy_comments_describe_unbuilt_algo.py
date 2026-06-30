# ACE-FP-EXPECT: clean
# CATEGORY: hallucination_bait
# LANGUAGE: python
# SOURCE: synthetic; comment-heavy file with a trivial real implementation
# WHY-CORRECT: the elaborate retrieval algorithm exists only in comments; the only executable code is a correct, trivial token counter — there is no LLM call, sink, or anti-pattern to flag
# EXPECTED-WRONG: citing the commented pseudo-algorithm as if it were real code (e.g. claiming an unbounded loop, missing retries, or an injection sink described in prose)
# CORRECT-VERDICT: no findings
"""Design notes for a future hybrid retrieval ranker.

NOTE: None of the algorithm below is implemented yet. This module ships only
`approx_token_count`. Everything else is a design sketch in comments.
"""

# ---------------------------------------------------------------------------
# PLANNED ALGORITHM: Hybrid Reciprocal-Rank-Fusion Retriever (NOT BUILT)
# ---------------------------------------------------------------------------
# Step 1. Accept a query string and an integer k.
# Step 2. Run two independent retrievers in parallel:
#           - a BM25 lexical retriever over the inverted index
#           - a dense retriever using cosine similarity over embeddings
# Step 3. For each retriever, produce a ranked list of document ids.
# Step 4. Fuse the two ranked lists with Reciprocal Rank Fusion:
#           score(d) = sum_over_retrievers( 1 / (rrf_k + rank_r(d)) )
#         where rrf_k is a smoothing constant (default 60).
# Step 5. If a query embedding call fails, retry up to 3 times with
#         exponential backoff, then fall back to lexical-only ranking.
# Step 6. De-duplicate documents that share a canonical_id.
# Step 7. Apply a recency boost: multiply score by (1 + 0.1 * recency_decay).
# Step 8. Truncate to the top k results and attach provenance metadata so the
#         downstream LLM can cite [doc_id] for every grounded claim.
# Step 9. Cache the fused result keyed by (query_hash, k) for 5 minutes.
#
# Open question: should step 4 weight the dense list higher for short queries?
# We will A/B test rrf_k in {30, 60, 90} once the retrievers exist.
# ---------------------------------------------------------------------------


def approx_token_count(text: str) -> int:
    """Rough whitespace token estimate (the only real code here).

    This is a deliberately simple heuristic, not a model tokenizer.
    """
    return len(text.split())
