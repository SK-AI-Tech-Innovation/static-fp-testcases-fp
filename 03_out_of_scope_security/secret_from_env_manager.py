# ACE-FP-EXPECT: scoped-out
# CATEGORY: 03_out_of_scope_security
# SOURCE: loading the Anthropic API key from env / a secret manager rather than hardcoding it
# WHY-CORRECT: secret handling is a SECURITY concern owned by the Sentinel engine. Static must stay silent
#              on secrets/keys. The code here is also security-correct: no literal key, value comes from the
#              environment (or a pluggable secret backend) and the client is constructed from it.
# EXPECTED-WRONG: a secret-exposure / key-management finding ("API key handling", "validate secret source",
#                 "rotate credentials") — security is out-of-scope; static defers to Sentinel.
# CORRECT-VERDICT: no findings (secret management is Sentinel's domain; static stays silent)
"""Construct the Anthropic client from an env/secret-manager-sourced key (no hardcoding)."""
from __future__ import annotations

import os

import anthropic


def _load_api_key() -> str:
    """Resolve the key from a secret manager if configured, else from the environment."""
    backend = os.environ.get("SECRET_BACKEND", "env")
    if backend == "vault":
        # Pluggable secret-manager hook; returns the resolved secret value.
        from secrets_client import read_secret  # type: ignore[import-not-found]

        return read_secret("anthropic/api_key")
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    return key


def make_client() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=_load_api_key())


def ask(prompt: str) -> str:
    client = make_client()
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in response.content if b.type == "text").strip()


if __name__ == "__main__":
    print(ask("Give me one productivity tip."))
