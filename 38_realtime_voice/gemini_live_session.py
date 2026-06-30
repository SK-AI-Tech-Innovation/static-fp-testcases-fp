# ACE-FP-EXPECT: clean
# CATEGORY: 38_realtime_voice
# SOURCE: Gemini Live API (google-genai) — client.aio.live.connect, send_realtime_input(audio=Blob)
# WHY-CORRECT: the Gemini Live API is a streaming bidirectional voice session. Audio goes in as a
#              Blob (audio/pcm;rate=16000) via send_realtime_input, and audio comes back on the
#              async response stream (24kHz out). There is no `.choices`, no chat-completion JSON to
#              parse, and structured output / response_format do not apply to a live audio session.
# EXPECTED-WRONG: a text-chat-centric engine fails to recognize the Live session and flags
#                 "AI call missing structured output / no response parsing" or "non-AI streaming code".
# CORRECT-VERDICT: no findings
"""Run a Gemini Live audio session: stream PCM in, receive audio out."""
from __future__ import annotations

import asyncio

from google import genai
from google.genai import types

MODEL = "gemini-3.1-flash-live-preview"


async def converse(pcm_frames_16k: list[bytes]) -> bytearray:
    """Send 16kHz PCM frames and collect the 24kHz audio reply."""
    client = genai.Client()
    config = types.LiveConnectConfig(response_modalities=["AUDIO"])
    out = bytearray()

    async with client.aio.live.connect(model=MODEL, config=config) as session:
        for frame in pcm_frames_16k:
            await session.send_realtime_input(
                audio=types.Blob(data=frame, mime_type="audio/pcm;rate=16000")
            )

        async for response in session.receive():
            if response.data is not None:
                out.extend(response.data)  # 24kHz PCM audio chunk
    return out


if __name__ == "__main__":
    asyncio.run(converse([]))
