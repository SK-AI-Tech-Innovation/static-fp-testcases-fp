# ACE-FP-EXPECT: clean
# CATEGORY: 47_phantom_bug_in_correct_code
# LANGUAGE: python
# SOURCE: ai-readable-data PR #1034 (inference_tools.py) — ACE: "history 파라미터가 str로 잘못 정의됨 — MCP 스키마 불일치"; author confirmed FP (str is correct for the MCP tool param)
# WHY-CORRECT: exposing `history` as a JSON-string parameter and deserializing it server-side with
#              json.loads is a deliberate, valid MCP tool-parameter shape — the tool takes a str and
#              parses it (with a type guard). It is not a "type mismatch"; the schema and the handler
#              agree that the wire type is str.
# EXPECTED-WRONG: engine assumes the param "should" be list[str] and flags a schema/type mismatch,
#                 not modeling that a str-encoded JSON param decoded with a guarded json.loads is an
#                 intentional, correct MCP tool idiom.
# CORRECT-VERDICT: no findings
"""An MCP tool that takes history as a JSON string and decodes it server-side.

ACE flagged `history: str` as a schema/type mismatch (claiming it should be list[str]).
The str wire type is the intended MCP tool-param shape; the handler guards and decodes it.
"""
import json


def ask_documents(question: str, history: str = "[]") -> dict:
    # `history` is intentionally a JSON string at the MCP boundary. Guard, then decode.
    parsed = json.loads(history) if isinstance(history, str) else []
    if not isinstance(parsed, list):
        parsed = []
    return {"question": question, "turns": len(parsed)}
