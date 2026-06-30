# ACE-FP-EXPECT: clean
# CATEGORY: 19_basic_correct_embeddings_and_misc
# SOURCE: OpenAI Python SDK (`openai`) `client.embeddings.create`
# WHY-CORRECT: textbook embedding call — model set, input passed as a list, vectors read from
#              data[i].embedding in order. Nothing about this call is incomplete or wrong.
# EXPECTED-WRONG: engine invents "batch the inputs", "add retry", or "normalize the vectors"
# CORRECT-VERDICT: no findings
"""Embed a list of texts with text-embedding-3-small."""
from openai import OpenAI

client = OpenAI()


def embed(texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    return [item.embedding for item in response.data]


if __name__ == "__main__":
    vectors = embed(["hello world", "goodbye world"])
    print(len(vectors), len(vectors[0]))
