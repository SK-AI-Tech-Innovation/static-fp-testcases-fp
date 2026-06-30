# ACE-FP-EXPECT: clean
# CATEGORY: 32_embedding_model_variety
# SOURCE: Jina AI embeddings HTTP API (`https://api.jina.ai/v1/embeddings`)
# WHY-CORRECT: Jina exposes an OpenAI-style embeddings endpoint; the request sends `model` and
#              `input`, authenticates with a bearer token, and reads vectors from `data[i].embedding`
#              in order. The payload and response parsing match Jina's documented API exactly.
# EXPECTED-WRONG: dated skill pack flags "raw requests call — use the official client" or doesn't
#                 recognize jina-embeddings and calls the model name invalid.
# CORRECT-VERDICT: no findings
"""Embed texts via the Jina AI embeddings REST API."""
import os

import requests

JINA_URL = "https://api.jina.ai/v1/embeddings"


def embed(texts: list[str]) -> list[list[float]]:
    response = requests.post(
        JINA_URL,
        headers={
            "Authorization": f"Bearer {os.environ['JINA_API_KEY']}",
            "Content-Type": "application/json",
        },
        json={"model": "jina-embeddings-v3", "input": texts},
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()["data"]
    return [item["embedding"] for item in data]


if __name__ == "__main__":
    vectors = embed(["hello", "world"])
    print(len(vectors), len(vectors[0]))
