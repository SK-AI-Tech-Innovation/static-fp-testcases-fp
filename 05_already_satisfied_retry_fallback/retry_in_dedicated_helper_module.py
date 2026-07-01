# ACE-FP-EXPECT: clean
# CATEGORY: 05_already_satisfied_retry_fallback
# LANGUAGE: python
# SOURCE: ai-readable-data PR #367 (serialize/node.py) — ACE: "직렬화에 재시도/폴백 누락"; author confirmed FP (retry lives in helper)
# WHY-CORRECT: the serialize call delegates to retry_transient_upload(), a tenacity-based helper that
#              retries transient 5xx/transport/timeout errors, and the node wraps the stage in
#              try/except for a fallback. Retry + fallback ARE present — just factored into a helper
#              rather than inlined at this call site.
# EXPECTED-WRONG: engine looks only at this call site, sees no inline tenacity/try-loop, and flags
#                 "missing retry/fallback" — missing that resilience is provided by the called helper.
# CORRECT-VERDICT: no findings
"""A serialize node whose retry/fallback live in a dedicated helper, not inlined.

ACE flagged missing retry/fallback at the serialize call. The helper it delegates to is
tenacity-backed, and the stage is wrapped for fallback — resilience is present.
"""
from typing import Any

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential


class TransientUploadError(Exception):
    ...


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=1, max=8),
    retry=retry_if_exception_type(TransientUploadError),
)
def retry_transient_upload(mgr: Any, payload: Any, graph_uri: str) -> str:
    # Tenacity retries transient failures here, in the shared helper.
    return mgr.serialize(payload, graph_uri=graph_uri)


def serialize_node(state: dict, mgr: Any) -> dict:
    try:
        # Delegates to the retrying helper above (bounded retry + backoff).
        result = retry_transient_upload(mgr, state["artifact"], state["graph_uri"])
        return {"serialized": result}
    except TransientUploadError:
        # Fallback: record failure instead of crashing the pipeline.
        return {"serialized": None, "status": "FAILED"}
