# ACE-FP-EXPECT: clean
# CATEGORY: 07_non_ai_false_detection
# SOURCE: JWT-style auth token verification + a tiny arithmetic lexer (compiler tokens)
# WHY-CORRECT: "token" appears in two non-AI senses: (1) a signed authentication token
#              verified with HMAC, and (2) lexical tokens produced by a tokenizer/lexer for
#              an expression language. Neither is an LLM context/usage token; there is no
#              tokenizer for a model and no token-count budgeting for prompts.
# EXPECTED-WRONG: keyword "token" / "Tokenizer" / "count_tokens" -> false "LLM token usage"
#                 detection -> spurious "no max_tokens set / token limit" findings.
# CORRECT-VERDICT: no findings
"""Auth token (HMAC) verification and a small arithmetic lexer. No LLM tokens."""
from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass


def sign_token(payload: str, secret: bytes) -> str:
    """Produce a hex HMAC signature acting as an authentication token."""
    return hmac.new(secret, payload.encode("utf-8"), hashlib.sha256).hexdigest()


def verify_token(payload: str, token: str, secret: bytes) -> bool:
    """Constant-time compare a presented auth token against the expected signature."""
    expected = sign_token(payload, secret)
    return hmac.compare_digest(expected, token)


@dataclass
class Token:
    """A lexical token from the arithmetic tokenizer (kind + literal text)."""

    kind: str
    text: str


def tokenize(source: str) -> list[Token]:
    """Split an arithmetic expression into lexical tokens (numbers and operators)."""
    tokens: list[Token] = []
    i = 0
    while i < len(source):
        ch = source[i]
        if ch.isspace():
            i += 1
            continue
        if ch.isdigit():
            j = i
            while j < len(source) and source[j].isdigit():
                j += 1
            tokens.append(Token("NUMBER", source[i:j]))
            i = j
            continue
        if ch in "+-*/()":
            tokens.append(Token("OP", ch))
            i += 1
            continue
        raise ValueError(f"unexpected character {ch!r} at position {i}")
    return tokens


def count_tokens(source: str) -> int:
    """Return how many lexical tokens an expression contains."""
    return len(tokenize(source))
