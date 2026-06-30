# ACE-FP-EXPECT: clean
# CATEGORY: 32_embedding_model_variety
# SOURCE: sentence-transformers with e5 / INSTRUCTOR-style instruction prefixes
# WHY-CORRECT: e5 and INSTRUCTOR family models REQUIRE asymmetric prefixes — "query:" for queries
#              and "passage:" for documents — to embed into the right space. Applying these
#              prefixes before encoding is mandatory and done correctly here. Omitting them would
#              be the actual bug; this code is complete.
# EXPECTED-WRONG: dated skill pack flags the literal "query:"/"passage:" strings as "stray prompt
#                 text leaking into the input" or "magic string", not knowing they are required.
# CORRECT-VERDICT: no findings
"""Embed queries and passages with intfloat/e5-large-v2 using the required prefixes."""
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("intfloat/e5-large-v2")


def embed_passages(passages: list[str]) -> list[list[float]]:
    prefixed = [f"passage: {p}" for p in passages]
    return model.encode(prefixed, normalize_embeddings=True).tolist()


def embed_query(query: str) -> list[float]:
    return model.encode([f"query: {query}"], normalize_embeddings=True)[0].tolist()


if __name__ == "__main__":
    docs = embed_passages(["Paris is the capital of France."])
    q = embed_query("What is the capital of France?")
    print(len(docs), len(q))
