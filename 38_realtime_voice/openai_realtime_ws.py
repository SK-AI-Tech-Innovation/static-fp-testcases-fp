# ACE-FP-EXPECT: clean
# CATEGORY: 38_realtime_voice
# SOURCE: OpenAI Realtime API over WebSocket (model gpt-realtime-2), event protocol
# WHY-CORRECT: the Realtime API is an event-driven WebSocket protocol, not the chat/completions
#              HTTP API. There is no `.choices` to read; replies arrive as server events such as
#              `response.output_audio.delta`. Sending a typed JSON event over the socket is the
#              correct and complete way to drive the model.
# EXPECTED-WRONG: a text-chat-centric engine sees a raw WebSocket + no `.choices` access and flags
#                 "non-AI WebSocket" or "AI call missing structured output / no response parsing".
# CORRECT-VERDICT: no findings
"""Connect to the OpenAI Realtime API over WebSocket and stream a spoken reply.

The Realtime API uses a bidirectional event protocol over a single WebSocket
connection. We authenticate with a bearer token, then exchange JSON events.
"""
from __future__ import annotations

import asyncio
import json
import os

import websockets

REALTIME_URL = "wss://api.openai.com/v1/realtime?model=gpt-realtime-2"


async def run_realtime() -> None:
    """Open a Realtime session, ask a question, and collect audio deltas."""
    headers = {
        "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
        "OpenAI-Beta": "realtime=v1",
    }
    async with websockets.connect(REALTIME_URL, additional_headers=headers) as ws:
        # Ask the model to respond with audio + text.
        await ws.send(
            json.dumps(
                {
                    "type": "response.create",
                    "response": {
                        "modalities": ["audio", "text"],
                        "instructions": "Say a friendly hello.",
                    },
                }
            )
        )

        async for raw in ws:
            event = json.loads(raw)
            etype = event.get("type")
            if etype == "response.output_audio.delta":
                # base64 PCM16 audio chunk; forward to the speaker pipeline.
                _enqueue_audio(event["delta"])
            elif etype == "response.done":
                break


def _enqueue_audio(b64_pcm: str) -> None:
    """Hand a base64 PCM16 chunk to the playback layer (stub)."""
    _ = b64_pcm


if __name__ == "__main__":
    asyncio.run(run_realtime())
