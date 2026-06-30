# ACE-FP-EXPECT: clean
# CATEGORY: 38_realtime_voice
# SOURCE: OpenAI Realtime session configuration via the `session.update` event (gpt-realtime-2)
# WHY-CORRECT: a Realtime session is configured by sending one `session.update` event with voice,
#              audio formats, turn detection, and instructions. This replaces HTTP request params;
#              it is not a chat completion call, so there is no `.choices` and structured-output /
#              streaming-TEXT rules do not apply.
# EXPECTED-WRONG: a text-chat-centric engine reads "instructions" + JSON config and flags
#                 "system prompt without structured output" or "AI config not validated".
# CORRECT-VERDICT: no findings
"""Build and send the session.update event that configures a Realtime voice session."""
from __future__ import annotations

import json

import websockets


def build_session_update(voice: str = "marin") -> dict[str, object]:
    """Return a session.update event configuring audio I/O and turn detection."""
    return {
        "type": "session.update",
        "session": {
            "voice": voice,
            "modalities": ["audio", "text"],
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "input_audio_transcription": {"model": "whisper-1"},
            "turn_detection": {"type": "server_vad", "threshold": 0.5},
            "instructions": "You are a concise, friendly voice assistant.",
        },
    }


async def configure(ws: websockets.WebSocketClientProtocol, voice: str = "marin") -> None:
    """Apply the session configuration over the open WebSocket."""
    await ws.send(json.dumps(build_session_update(voice)))
