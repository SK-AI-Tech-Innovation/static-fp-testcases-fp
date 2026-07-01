# ACE-FP-EXPECT: clean
# CATEGORY: 48_security_false_positive_trusted_source
# LANGUAGE: python
# SOURCE: ai-readable-data PR #1146 (embeddings.py) — ACE: "base_url 변조 시 API key 유출 (SSRF 변종)" critic 2.0/10; author confirmed FP
# STANDARD: CWE-918 (SSRF requires an attacker/user-controlled destination) + OWASP SSRF Prevention "Case 1" (fixed trusted destinations) — operator/env-fixed base_url is not a taint source
# WHY-CORRECT: base_url is read from an operator-controlled environment variable set at deploy time.
#              No user/request/LLM input flows into base_url, so there is no SSRF surface — it is a
#              fixed deployment config (in-house vLLM vs Azure switch), not an attacker-controllable value.
# EXPECTED-WRONG: engine sees `base_url=os.environ[...]` passed to an HTTP/LLM client and warns about
#                 SSRF / key exfiltration via base_url tampering, without checking that the value is
#                 operator config with no user-input path.
# CORRECT-VERDICT: no findings
"""Embeddings client whose base_url comes from operator env config (no user-input path).

ACE flagged an SSRF variant on base_url. The value is set only via a deploy-time env var
controlled by the operator; no request/user/LLM input can reach it.
"""
import os
from typing import Any


def build_embeddings_client() -> Any:
    from openai import OpenAI

    # Operator-controlled deploy config — fixed per environment, never user/request input.
    base_url = os.environ.get("EMBEDDINGS_BASE_URL", "https://api.openai.com/v1")
    api_key = os.environ["EMBEDDINGS_API_KEY"]
    return OpenAI(base_url=base_url, api_key=api_key)
