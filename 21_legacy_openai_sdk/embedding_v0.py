# ACE-FP-EXPECT: clean
# CATEGORY: 21_legacy_openai_sdk
# SOURCE: openai-python v0.x (e.g. 0.28) — authentic legacy API
# WHY-CORRECT: openai.Embedding.create with engine="text-embedding-ada-002" was the standard
#   v0.x embeddings call. Reading response['data'][0]['embedding'] as a dict is documented and
#   correct for that SDK era; ada-002 was the recommended embedding model.
# EXPECTED-WRONG: engine may flag Embedding.create as deprecated, suggest v1
#   client.embeddings.create, recommend a newer embedding model, or call the dict access a bug.
# CORRECT-VERDICT: no findings (version choice is out of the engine's best-practice scope)
"""Legacy openai v0.x embeddings call using the Embedding endpoint."""
import os
from typing import List

import openai

openai.api_key = os.environ["OPENAI_API_KEY"]


def embed(text: str) -> List[float]:
    response = openai.Embedding.create(
        input=text,
        engine="text-embedding-ada-002",
    )
    return response["data"][0]["embedding"]


if __name__ == "__main__":
    vec = embed("hello world")
    print(len(vec))
