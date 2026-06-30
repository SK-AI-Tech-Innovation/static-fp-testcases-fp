# ACE-FP-EXPECT: clean
# CATEGORY: 25_multimodal
# SOURCE: OpenAI Python SDK (`openai`) text-to-speech with gpt-4o-mini-tts
# WHY-CORRECT: correct TTS usage — audio.speech.create takes a model, a voice, and input text, and
#              returns binary audio streamed to a file. There is no chat completion, no token limit,
#              and no structured output involved; this is the documented synthesis shape.
# EXPECTED-WRONG: engine treats the `input` text as a prompt and suggests "add a system message",
#                 "set max_tokens", or "request structured JSON output" on a speech call.
# CORRECT-VERDICT: no findings
"""Synthesize speech from text and write it to an MP3 file."""
from pathlib import Path

from openai import OpenAI

client = OpenAI()


def speak(text: str, out_path: str = "speech.mp3") -> str:
    response = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=text,
    )
    response.stream_to_file(Path(out_path))
    return out_path


if __name__ == "__main__":
    print(speak("Hello there, this is a synthesized voice."))
