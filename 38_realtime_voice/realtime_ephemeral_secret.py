# ACE-FP-EXPECT: clean
# CATEGORY: 38_realtime_voice
# SOURCE: Server-side minting of an ephemeral Realtime client secret for browser WebRTC sessions
# WHY-CORRECT: browsers must never hold the real API key; the backend exchanges it for a short-lived
#              ephemeral client secret via the Realtime sessions endpoint and returns only that to the
#              client. This is a token-minting endpoint, not a chat completion — no `.choices`, no
#              structured-output expectation, and the single HTTP POST does not need retry/streaming logic.
# EXPECTED-WRONG: a text-chat-centric engine sees an OpenAI HTTP call without `.choices` parsing and
#                 flags "AI call missing structured output" or "no retry/fallback on AI request".
# CORRECT-VERDICT: no findings
"""Mint a short-lived ephemeral client secret for a browser Realtime session."""
from __future__ import annotations

import os

import httpx

SESSIONS_URL = "https://api.openai.com/v1/realtime/sessions"


def mint_ephemeral_secret() -> dict[str, object]:
    """Create an ephemeral Realtime session token for the browser to use.

    Returns the JSON the frontend needs (it contains `client_secret.value`),
    keeping the long-lived API key on the server only.
    """
    resp = httpx.post(
        SESSIONS_URL,
        headers={
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
            "Content-Type": "application/json",
        },
        json={"model": "gpt-realtime-2", "voice": "marin"},
        timeout=10.0,
    )
    resp.raise_for_status()
    return resp.json()
