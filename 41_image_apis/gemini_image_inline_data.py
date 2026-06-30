# ACE-FP-EXPECT: clean
# CATEGORY: 41_image_apis
# SOURCE: Gemini image generation ("Nano Banana") — generate_content(model="gemini-2.5-flash-image")
# WHY-CORRECT: Gemini returns a generated image as an `inline_data` part inside the response's
#              candidates/content/parts, not as `.choices` and not as text. Iterating parts to find
#              `inline_data` is the correct way to extract the image; structured-output / response_format
#              rules for text chat do not apply to an image-out generation.
# EXPECTED-WRONG: a text-chat-centric engine expects `.text`/`.choices` and flags "AI response not parsed",
#                 "missing structured output", or "no text extracted from model response".
# CORRECT-VERDICT: no findings
"""Generate an image with Gemini and extract it from the inline_data part."""
from __future__ import annotations

from google import genai

MODEL = "gemini-2.5-flash-image"


def generate(prompt: str) -> bytes:
    """Return raw image bytes produced by Gemini for `prompt`."""
    client = genai.Client()
    response = client.models.generate_content(model=MODEL, contents=[prompt])
    for part in response.candidates[0].content.parts:
        if part.inline_data is not None:
            return part.inline_data.data  # raw image bytes
    raise RuntimeError("model returned no image part")


def edit(prompt: str, source_image: bytes) -> bytes:
    """Edit an existing image by passing it alongside the text prompt."""
    client = genai.Client()
    response = client.models.generate_content(
        model=MODEL,
        contents=[prompt, {"inline_data": {"mime_type": "image/png", "data": source_image}}],
    )
    for part in response.candidates[0].content.parts:
        if part.inline_data is not None:
            return part.inline_data.data
    raise RuntimeError("model returned no image part")
