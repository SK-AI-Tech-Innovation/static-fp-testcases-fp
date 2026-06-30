# ACE-FP-EXPECT: clean
# CATEGORY: 32_embedding_model_variety
# SOURCE: Voyage AI Python SDK (`voyageai`) `client.embed`
# WHY-CORRECT: Voyage's client returns embeddings under `result.embeddings`, and v3 models accept
#              an `input_type` of "document" or "query". Both calls set the model and input_type
#              correctly and read the vectors as documented. Complete and idiomatic Voyage usage.
# EXPECTED-WRONG: dated skill pack has never seen `voyageai` and flags "unknown provider/SDK" or
#                 mistakes `input_type` for an invalid parameter.
# CORRECT-VERDICT: no findings
"""Embed documents and queries with Voyage AI voyage-3."""
import voyageai

client = voyageai.Client()
MODEL = "voyage-3"


def embed_documents(docs: list[str]) -> list[list[float]]:
    result = client.embed(docs, model=MODEL, input_type="document")
    return result.embeddings


def embed_query(query: str) -> list[float]:
    result = client.embed([query], model=MODEL, input_type="query")
    return result.embeddings[0]


if __name__ == "__main__":
    vectors = embed_documents(["first passage", "second passage"])
    print(len(vectors), len(vectors[0]))
