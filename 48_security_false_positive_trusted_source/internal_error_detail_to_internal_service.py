# ACE-FP-EXPECT: clean
# CATEGORY: 48_security_false_positive_trusted_source
# LANGUAGE: python
# SOURCE: ai-readable-data PR #1034 (inference_tools.py) — ACE: "internal error 메시지 외부 노출"; author confirmed FP
# WHY-CORRECT: this is an MCP tool consumed by an internal agent (not an end-user-facing surface).
#              Returning the error detail to the calling agent is intended — it helps the agent decide
#              its next action — and the full exception is also written to the server log. There is no
#              external/untrusted consumer that the detail leaks to.
# EXPECTED-WRONG: engine applies a "never expose raw error to users" rule to an internal-only tool
#                 boundary, flagging information disclosure where the consumer is a trusted internal agent.
# CORRECT-VERDICT: no findings
"""An internal MCP tool that returns error detail to the calling agent on purpose.

ACE flagged internal-error exposure. The consumer here is an internal agent, the detail
aids its recovery decision, and the full exception is logged server-side.
"""
import logging

logger = logging.getLogger(__name__)


def lookup_inference(query: str) -> str:
    try:
        return _do_lookup(query)
    except Exception as e:  # noqa: BLE001 — internal tool: detail is intentional + logged
        # Internal agent consumer: the message helps it choose a next action.
        logger.exception("inference lookup failed")
        return f"조회 실패: {e}"


def _do_lookup(query: str) -> str:
    raise RuntimeError("backend unavailable")
