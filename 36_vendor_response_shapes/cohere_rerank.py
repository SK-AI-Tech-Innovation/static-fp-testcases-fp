# ACE-FP-EXPECT: clean
# CATEGORY: 36_vendor_response_shapes
# SOURCE: Cohere ClientV2 rerank, verified June 2026
# WHY-CORRECT: Cohere rerank returns res.results as objects with .index and .relevance_score; there is no .choices and no OpenAI-style payload
# EXPECTED-WRONG: stale analyzer flags rerank as an unknown endpoint / expects .choices / flags res.results[*].relevance_score as malformed
# CORRECT-VERDICT: no findings
"""Rerank documents against a query with Cohere rerank-v3.5."""

import os

import cohere

co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))

DOCUMENTS = [
    "Carson City is the capital of Nevada.",
    "The capital of France is Paris.",
    "Washington, D.C. is the capital of the United States.",
]


def main() -> None:
    res = co.rerank(
        model="rerank-v3.5",
        query="What is the capital of the United States?",
        documents=DOCUMENTS,
        top_n=2,
    )

    for hit in res.results:
        print(hit.index, round(hit.relevance_score, 4), DOCUMENTS[hit.index])


if __name__ == "__main__":
    main()
