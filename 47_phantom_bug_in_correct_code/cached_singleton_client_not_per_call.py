# ACE-FP-EXPECT: clean
# CATEGORY: 47_phantom_bug_in_correct_code
# LANGUAGE: python
# SOURCE: ace PR #923 (reference_docs_store.py) — ACE: "매 호출마다 새 Qdrant 클라이언트 생성 (연결 누수)"; author confirmed FP
# WHY-CORRECT: get_async_qdrant_client() uses a module-level singleton cache and returns the same
#              client on every call, so no new connection is created per call. There is no leak or
#              per-call construction overhead.
# EXPECTED-WRONG: engine sees get_async_qdrant_client() invoked inside a request function and assumes
#                 a fresh client is built each call, flagging a connection leak / overhead — missing
#                 the module-level singleton cache.
# CORRECT-VERDICT: no findings
"""A retrieval helper that reuses a cached singleton Qdrant client.

ACE flagged per-call client creation / connection leak; in fact the client is created
once and cached at module level, then reused.
"""
from functools import lru_cache
from typing import Any


@lru_cache(maxsize=1)
def get_async_qdrant_client() -> Any:
    # Built once, cached for the process lifetime — every caller gets the same instance.
    from qdrant_client import AsyncQdrantClient

    return AsyncQdrantClient(url="http://qdrant:6333")


async def search_reference_docs(collection_name: str, query_vector: list[float]) -> list[dict]:
    # Reuses the cached singleton — no new client/connection per call.
    client = get_async_qdrant_client()
    hits = await client.search(collection_name=collection_name, query_vector=query_vector, limit=5)
    return [{"id": h.id, "score": h.score} for h in hits]
