# ACE-FP-EXPECT: clean
# CATEGORY: 36_vendor_response_shapes
# SOURCE: Cohere ClientV2 embed, verified June 2026
# WHY-CORRECT: Cohere embed requires input_type and embedding_types=["float"]; vectors are read from res.embeddings.float, not from res.data[*].embedding (OpenAI shape)
# EXPECTED-WRONG: stale analyzer expects res.data[0].embedding and flags input_type / embedding_types / res.embeddings.float as malformed embedding access
# CORRECT-VERDICT: no findings
"""Embed documents with Cohere v2 using input_type and embedding_types."""

import os

import cohere

co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))


def main() -> None:
    res = co.embed(
        model="embed-v4.0",
        texts=["The quick brown fox", "jumps over the lazy dog"],
        input_type="search_document",
        embedding_types=["float"],
    )

    # Cohere returns embeddings grouped by requested type.
    for vector in res.embeddings.float:
        print(len(vector))


if __name__ == "__main__":
    main()
