# ACE-FP-EXPECT: clean
# CATEGORY: 38_realtime_voice
# SOURCE: ElevenLabs Conversational AI WebSocket (convai/conversation?agent_id=..., agent_response_complete)
# WHY-CORRECT: ElevenLabs Conversational AI is an event-driven voice agent over WebSocket. Turns are
#              signalled by events like `agent_response_complete`; audio arrives as base64 chunks.
#              There is no `.choices`, no chat-completion HTTP response, and structured output rules
#              do not apply to a streaming voice agent.
# EXPECTED-WRONG: a text-chat-centric engine sees a raw WebSocket loop and flags "non-AI WebSocket"
#                 or "AI response not parsed / no structured output".
# CORRECT-VERDICT: no findings
"""Drive an ElevenLabs Conversational AI agent over WebSocket."""
from __future__ import annotations

import asyncio
import json
import os

import websockets


def conversation_url(agent_id: str) -> str:
    """Build the convai WebSocket URL for a given agent."""
    return f"wss://api.elevenlabs.io/v1/convai/conversation?agent_id={agent_id}"


async def talk(agent_id: str) -> None:
    """Open a conversation and process audio + turn-completion events."""
    headers = {"xi-api-key": os.environ["ELEVENLABS_API_KEY"]}
    async with websockets.connect(
        conversation_url(agent_id), additional_headers=headers
    ) as ws:
        async for raw in ws:
            event = json.loads(raw)
            etype = event.get("type")
            if etype == "audio":
                _play(event["audio_event"]["audio_base_64"])
            elif etype == "agent_response_complete":
                # The agent finished its turn; ready for the next user utterance.
                break


def _play(b64_audio: str) -> None:
    """Forward a base64 audio chunk to playback (stub)."""
    _ = b64_audio


if __name__ == "__main__":
    asyncio.run(talk(os.environ.get("AGENT_ID", "demo")))
