# ACE-FP-EXPECT: clean
# CATEGORY: 05_already_satisfied_retry_fallback
# LANGUAGE: python
# SOURCE: ai-readable-data PR #1034 (inference_tools.py / inference_client.py) — ACE: "AI Agent 추론 요청에 부족한/없는 HTTP 타임아웃"; author confirmed FP (REQUEST_TIMEOUT=600 already set)
# WHY-CORRECT: the HTTP client is constructed with an explicit, generous request timeout
#              (REQUEST_TIMEOUT=600s) sized for long agent inference. The call site inherits that
#              client-level timeout, so a per-call timeout= is not required — the timeout IS configured.
# EXPECTED-WRONG: engine sees a post()/request call with no inline timeout= kwarg and flags "no/
#                 insufficient HTTP timeout", not tracing that the shared client already sets an
#                 explicit timeout tuned for the workload.
# CORRECT-VERDICT: no findings
"""An HTTP client with an explicit, workload-sized timeout reused across calls.

ACE flagged "no/insufficient timeout" at the call site. The client is built once with
REQUEST_TIMEOUT=600s (tuned for long agent inference); the call inherits it.
"""
import os

import httpx

# Explicit, generous timeout sized for long KG-agent inference (not the ~5-10s default).
REQUEST_TIMEOUT = float(os.environ.get("REQUEST_TIMEOUT", "600"))

# Client-level timeout applies to every request made through this client.
_client = httpx.Client(base_url=os.environ["INFERENCE_URL"], timeout=REQUEST_TIMEOUT)


def post(endpoint: str, payload: dict) -> dict:
    # No inline timeout= needed: _client already carries REQUEST_TIMEOUT for every call.
    resp = _client.post(endpoint, json=payload)
    resp.raise_for_status()
    return resp.json()
