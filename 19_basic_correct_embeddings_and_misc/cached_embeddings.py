# ACE-FP-EXPECT: clean
# CATEGORY: 19_basic_correct_embeddings_and_misc
# SOURCE: OpenAI embeddings with a content-hash cache to skip recomputation
# WHY-CORRECT: each text is keyed by a stable sha256 of its bytes; cache hits are returned and only
#              misses are embedded, then stored. A clean, correct memoization of an expensive call.
# EXPECTED-WRONG: engine flags "no cache invalidation/TTL" — irrelevant since embeddings are pure
#                 functions of (model, text); the key already pins the content.
# CORRECT-VERDICT: no findings
"""Embed texts with a content-hash cache to avoid recomputing vectors."""
import hashlib

from openai import OpenAI

client = OpenAI()
_cache: dict[str, list[float]] = {}


def _key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def embed_cached(text: str) -> list[float]:
    key = _key(text)
    if key in _cache:
        return _cache[key]
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=[text],
    )
    vector = response.data[0].embedding
    _cache[key] = vector
    return vector


if __name__ == "__main__":
    a = embed_cached("repeated text")
    b = embed_cached("repeated text")
    print(a is b)
