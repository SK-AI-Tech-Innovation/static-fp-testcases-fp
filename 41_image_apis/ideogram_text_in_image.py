# ACE-FP-EXPECT: clean
# CATEGORY: 41_image_apis
# SOURCE: Ideogram image generation REST API (strong text-in-image rendering)
# WHY-CORRECT: Ideogram's generate endpoint returns image descriptors (URLs) on a `data` array from a
#              single REST call. It is an image API: no `.choices`, no chat messages, no structured-output
#              contract. The `prompt` is an image prompt and "text" here means glyphs rendered INTO the
#              image, not a chat text response to validate.
# EXPECTED-WRONG: a text-chat-centric engine sees "prompt"/"text" + an AI vendor and flags "AI call
#                 missing structured output", "no .choices", or "model text output not parsed".
# CORRECT-VERDICT: no findings
"""Generate an image with rendered text using the Ideogram API."""
from __future__ import annotations

import os

import httpx

GENERATE_URL = "https://api.ideogram.ai/v1/ideogram-v3/generate"


def generate_with_text(prompt: str) -> str:
    """Return the URL of a generated image (which renders text per `prompt`)."""
    resp = httpx.post(
        GENERATE_URL,
        headers={"Api-Key": os.environ["IDEOGRAM_API_KEY"]},
        data={"prompt": prompt, "rendering_speed": "QUALITY", "aspect_ratio": "16x9"},
        timeout=120.0,
    )
    resp.raise_for_status()
    # Result images are listed under `data`, not `.choices`.
    return resp.json()["data"][0]["url"]


if __name__ == "__main__":
    print(generate_with_text('A poster that says "GRAND OPENING" in bold retro type'))
