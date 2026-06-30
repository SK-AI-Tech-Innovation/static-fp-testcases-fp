# ACE-FP-EXPECT: clean
# CATEGORY: 41_image_apis
# SOURCE: Stability AI — Stable Diffusion 3.5 generate endpoint (REST, returns image bytes)
# WHY-CORRECT: Stability's generate endpoint returns image bytes (or base64) directly from a single
#              REST call. It is an image API: there is no `.choices`, no chat message array, no system
#              prompt, and no structured-output / streaming-TEXT contract. `prompt` here is an image prompt.
# EXPECTED-WRONG: a text-chat-centric engine sees "prompt" + an AI vendor and flags "LLM call missing
#                 structured output", "no .choices parsing", or "prompt template / injection" concerns.
# CORRECT-VERDICT: no findings
"""Generate an image with Stable Diffusion 3.5 via the Stability AI REST API."""
from __future__ import annotations

import os

import httpx

GENERATE_URL = "https://api.stability.ai/v2beta/stable-image/generate/sd3"


def generate(prompt: str, aspect_ratio: str = "1:1") -> bytes:
    """Return raw PNG bytes for `prompt` from Stable Diffusion 3.5."""
    resp = httpx.post(
        GENERATE_URL,
        headers={
            "Authorization": f"Bearer {os.environ['STABILITY_API_KEY']}",
            "Accept": "image/*",
        },
        files={"none": ""},
        data={"prompt": prompt, "model": "sd3.5-large", "aspect_ratio": aspect_ratio},
        timeout=120.0,
    )
    resp.raise_for_status()
    return resp.content  # raw image bytes


if __name__ == "__main__":
    img = generate("a neon city skyline at dusk")
    print(len(img), "bytes")
