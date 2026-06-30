# ACE-FP-EXPECT: clean
# CATEGORY: 25_multimodal
# SOURCE: OpenAI Python SDK (`openai`) audio transcription with whisper-1
# WHY-CORRECT: correct speech-to-text usage — audio.transcriptions.create takes model="whisper-1"
#              and a binary file handle. Transcript read from response.text. There is no chat
#              messages list and no token limit to set; this is the complete documented shape.
# EXPECTED-WRONG: engine mistakes this for a chat call and flags "missing messages", "missing
#                 system prompt", or "use structured output" on a transcription request.
# CORRECT-VERDICT: no findings
"""Transcribe an audio file to text with Whisper."""
from openai import OpenAI

client = OpenAI()


def transcribe(audio_path: str) -> str:
    with open(audio_path, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
        )
    return response.text


if __name__ == "__main__":
    print(transcribe("meeting.mp3"))
