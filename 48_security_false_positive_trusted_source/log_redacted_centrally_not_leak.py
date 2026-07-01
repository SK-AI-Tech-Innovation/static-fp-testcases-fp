# ACE-FP-EXPECT: clean
# CATEGORY: 48_security_false_positive_trusted_source
# LANGUAGE: python
# SOURCE: ai-readable-data PR #387 (agent.py) — ACE: "LLM 응답이 로그에 무조건 기록 (secret leakage)"; author confirmed FP
# WHY-CORRECT: the raw response is not logged at the call site — it is passed to a central
#              dump_diagnostic_context helper that truncates and redacts secrets before writing. The
#              sensitive-exposure path the engine describes does not exist.
# EXPECTED-WRONG: engine sees the response handed to a logging/diagnostic helper and assumes raw
#                 sensitive content reaches the log, flagging secret/PII leakage — missing that the
#                 helper redacts + truncates centrally.
# CORRECT-VERDICT: no findings
"""Diagnostics that route the LLM response through a redacting helper, not raw logging.

ACE flagged unconditional logging of the LLM response. The call site never logs it
directly; a central helper redacts secrets and truncates before anything is written.
"""
import logging
import re

logger = logging.getLogger(__name__)

_SECRET_RE = re.compile(r"(sk-[A-Za-z0-9]{8,}|Bearer\s+\S+)")


def _redact_secrets(text: str) -> str:
    return _SECRET_RE.sub("[REDACTED]", text)


def dump_diagnostic_context(response_text: str) -> None:
    # Central helper: redact secrets, then truncate — raw content never hits the log.
    safe = _redact_secrets(response_text)[:200]
    logger.warning("llm diagnostic: %s", safe)


def handle_response(response_text: str) -> str:
    # Call site does NOT log raw content; it delegates to the redacting helper.
    dump_diagnostic_context(response_text)
    return response_text
