# ACE-FP-EXPECT: clean
# CATEGORY: 41_image_apis
# SOURCE: OpenAI Images edit API — client.images.edit(image=[...], prompt=...) with multiple inputs
# WHY-CORRECT: images.edit composes one or more input images plus a prompt and returns the result on
#              `result.data[0].b64_json`. It is an image endpoint: no `.choices`, no chat messages, no
#              response_format / structured output, and a single call needs no streaming-TEXT handling.
# EXPECTED-WRONG: a text-chat-centric engine treats this image-gen call as a chat completion and flags
#                 "no .choices", "missing structured output", or "AI call without retry/fallback".
# CORRECT-VERDICT: no findings
"""Edit/compose multiple input images into one result via the OpenAI Images edit API."""
from __future__ import annotations

from openai import OpenAI

client = OpenAI()


def compose(prompt: str, image_paths: list[str]) -> str:
    """Blend several source images per `prompt`; return base64 PNG."""
    images = [open(path, "rb") for path in image_paths]
    try:
        result = client.images.edit(
            model="gpt-image-2",
            image=images,
            prompt=prompt,
            size="1024x1024",
        )
    finally:
        for f in images:
            f.close()
    return result.data[0].b64_json


if __name__ == "__main__":
    compose("Put the cat on the sofa", ["cat.png", "sofa.png"])
