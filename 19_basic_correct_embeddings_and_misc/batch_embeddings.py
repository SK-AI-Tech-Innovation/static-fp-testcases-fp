# ACE-FP-EXPECT: clean
# CATEGORY: 19_basic_correct_embeddings_and_misc
# SOURCE: OpenAI Python SDK (`openai`) `client.embeddings.create` with manual batching
# WHY-CORRECT: many texts are split into fixed-size batches that respect the API input limit,
#              each batch embedded in one call, and results concatenated in original order.
#              This is the recommended way to embed large corpora.
# EXPECTED-WRONG: engine claims "no batching" or "may exceed token limits" despite the explicit batch loop
# CORRECT-VERDICT: no findings
"""Embed many texts in batches that respect the API's input-size limit."""
from openai import OpenAI

client = OpenAI()

BATCH_SIZE = 100  # well within the embeddings input array limit


def embed_all(texts: list[str]) -> list[list[float]]:
    vectors: list[list[float]] = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=batch,
        )
        vectors.extend(item.embedding for item in response.data)
    return vectors


if __name__ == "__main__":
    corpus = [f"document number {i}" for i in range(250)]
    result = embed_all(corpus)
    print(len(result))
