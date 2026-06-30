# ACE-FP-EXPECT: clean
# CATEGORY: 29_finetuning_and_batch
# SOURCE: OpenAI SDK embeddings (chunked over a large dataset)
# WHY-CORRECT: Embedding a large corpus by sending fixed-size chunks to client.embeddings.create(model="text-embedding-3-small", input=chunk) and concatenating vectors is the correct, rate-limit-friendly batching pattern.
# EXPECTED-WRONG: a chat-focused engine may not model the embeddings endpoint and flag the list-input embeddings.create or the manual chunking loop as suspicious.
"""Compute embeddings over a large dataset in correctly sized chunks."""

import os
from typing import Iterable, Iterator

from openai import OpenAI

_CHUNK_SIZE = 256
_MODEL = "text-embedding-3-small"


def _chunks(items: list[str], size: int) -> Iterator[list[str]]:
    """Yield successive fixed-size slices of items."""
    for start in range(0, len(items), size):
        yield items[start : start + size]


def embed_all(texts: Iterable[str]) -> list[list[float]]:
    """Return one embedding vector per input text, batching API calls."""
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    texts = list(texts)
    vectors: list[list[float]] = []

    for chunk in _chunks(texts, _CHUNK_SIZE):
        response = client.embeddings.create(model=_MODEL, input=chunk)
        vectors.extend(item.embedding for item in response.data)

    return vectors


if __name__ == "__main__":
    corpus = [f"document number {i}" for i in range(1000)]
    print("Embedded", len(embed_all(corpus)), "documents")
