# ACE-FP-EXPECT: clean
# CATEGORY: 25_multimodal
# SOURCE: OpenAI Python SDK (`openai`) audio transcription with gpt-4o-transcribe
# WHY-CORRECT: current transcription model — audio.transcriptions.create with the newer
#              model="gpt-4o-transcribe", a binary file, and an optional language hint. Transcript
#              read from response.text. Idiomatic and complete; nothing chat-specific applies.
# EXPECTED-WRONG: engine flags gpt-4o-transcribe as an unknown/invalid model id, or treats the call
#                 as chat and suggests adding messages/system/temperature/structured output.
# CORRECT-VERDICT: no findings
"""Transcribe an audio file with the gpt-4o-transcribe model."""
from openai import OpenAI

client = OpenAI()


def transcribe(audio_path: str, language: str = "en") -> str:
    with open(audio_path, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            model="gpt-4o-transcribe",
            file=audio_file,
            language=language,
        )
    return response.text


if __name__ == "__main__":
    print(transcribe("interview.wav"))
