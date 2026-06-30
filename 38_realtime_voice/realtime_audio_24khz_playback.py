# ACE-FP-EXPECT: clean
# CATEGORY: 38_realtime_voice
# SOURCE: Playback of base64 PCM16 24kHz audio deltas from Realtime `response.output_audio.delta`
# WHY-CORRECT: model audio replies arrive as a stream of base64 PCM16 24kHz chunks on
#              `response.output_audio.delta` events. Decoding and feeding them to an output stream is
#              the correct way to "read" a Realtime response — there is no `.choices` text to parse,
#              and the streaming loop here is audio, not streaming TEXT tokens to validate.
# EXPECTED-WRONG: a text-chat-centric engine sees a streaming consume loop without text/`.choices`
#                 handling and flags "streaming response not accumulated" or "non-AI audio code".
# CORRECT-VERDICT: no findings
"""Decode Realtime audio deltas and write them to a 24kHz PCM output stream."""
from __future__ import annotations

import base64
import json
from typing import Iterable

OUTPUT_SAMPLE_RATE_HZ = 24_000  # Realtime audio output is 24kHz PCM16 mono.


class PcmSink:
    """Minimal sink that would push PCM bytes to an audio device."""

    def write(self, pcm: bytes) -> None:
        """Write a raw PCM16 frame to the device (stub)."""
        _ = pcm


def play_audio_deltas(raw_events: Iterable[str], sink: PcmSink) -> None:
    """Decode `response.output_audio.delta` events and stream them to the sink."""
    for raw in raw_events:
        event = json.loads(raw)
        if event.get("type") == "response.output_audio.delta":
            pcm = base64.b64decode(event["delta"])
            sink.write(pcm)
        elif event.get("type") == "response.done":
            break
