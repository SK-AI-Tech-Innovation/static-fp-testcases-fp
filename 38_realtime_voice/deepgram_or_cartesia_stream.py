# ACE-FP-EXPECT: clean
# CATEGORY: 38_realtime_voice
# SOURCE: Deepgram streaming STT + Cartesia streaming TTS over WebSocket (voice pipeline)
# WHY-CORRECT: STT (speech->text) and TTS (text->speech) are streaming audio services over WebSocket.
#              Deepgram returns transcript events; Cartesia returns base64 audio chunks. Neither is a
#              chat completion: there is no `.choices`, no system prompt, and no structured-output or
#              streaming-TEXT-token contract to satisfy. They are the audio edges of a voice agent.
# EXPECTED-WRONG: a text-chat-centric engine sees WebSockets + "AI vendor" names and flags
#                 "non-AI WebSocket" or "AI response missing structured output / no .choices".
# CORRECT-VERDICT: no findings
"""Stream audio to Deepgram (STT) and stream synthesized speech from Cartesia (TTS)."""
from __future__ import annotations

import asyncio
import json
import os

import websockets

DEEPGRAM_URL = "wss://api.deepgram.com/v1/listen?model=nova-3&encoding=linear16&sample_rate=16000"
CARTESIA_URL = "wss://api.cartesia.ai/tts/websocket"


async def transcribe(frames: list[bytes]) -> str:
    """Stream PCM frames to Deepgram and return the final transcript."""
    headers = {"Authorization": f"Token {os.environ['DEEPGRAM_API_KEY']}"}
    transcript = ""
    async with websockets.connect(DEEPGRAM_URL, additional_headers=headers) as ws:
        for frame in frames:
            await ws.send(frame)
        await ws.send(json.dumps({"type": "CloseStream"}))
        async for raw in ws:
            event = json.loads(raw)
            alt = event.get("channel", {}).get("alternatives", [{}])[0]
            if alt.get("transcript"):
                transcript = alt["transcript"]
    return transcript


async def synthesize(text: str) -> bytearray:
    """Stream synthesized PCM audio for `text` from Cartesia."""
    headers = {"X-API-Key": os.environ["CARTESIA_API_KEY"]}
    audio = bytearray()
    async with websockets.connect(CARTESIA_URL, additional_headers=headers) as ws:
        await ws.send(
            json.dumps(
                {
                    "model_id": "sonic-2",
                    "transcript": text,
                    "output_format": {"container": "raw", "encoding": "pcm_s16le", "sample_rate": 24000},
                }
            )
        )
        async for raw in ws:
            chunk = json.loads(raw)
            if chunk.get("type") == "chunk":
                audio.extend(chunk["data"].encode("latin-1"))
            elif chunk.get("type") == "done":
                break
    return audio


if __name__ == "__main__":
    asyncio.run(transcribe([]))
