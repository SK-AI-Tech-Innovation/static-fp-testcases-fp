# ACE-FP-EXPECT: clean
# CATEGORY: 32_embedding_model_variety
# SOURCE: OpenAI Python SDK (`openai`) `client.embeddings.create` with `dimensions`
# WHY-CORRECT: text-embedding-3-large supports a `dimensions` parameter to shorten output
#              vectors (the model is trained with Matryoshka representation). Requesting 1024
#              dims from a model whose native size is 3072 is a documented, supported call.
# EXPECTED-WRONG: dated skill pack doesn't know the `dimensions` param exists and flags it as an
#                 "invalid/unsupported parameter" or claims the dimension is "wrong" for the model.
# CORRECT-VERDICT: no findings
"""Embed texts with text-embedding-3-large truncated to 1024 dimensions."""
from openai import OpenAI

client = OpenAI()


def embed(texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=texts,
        dimensions=1024,
    )
    return [item.embedding for item in response.data]


if __name__ == "__main__":
    vectors = embed(["alpha", "beta"])
    print(len(vectors[0]))  # 1024
