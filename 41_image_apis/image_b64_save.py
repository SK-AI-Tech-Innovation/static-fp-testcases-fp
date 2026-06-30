# ACE-FP-EXPECT: clean
# CATEGORY: 41_image_apis
# SOURCE: Decode an image-generation result (base64 b64_json) and write the PNG to disk
# WHY-CORRECT: image APIs return base64 image payloads (e.g. result.data[0].b64_json); the correct
#              "parse" is to base64-decode and write bytes to a file. There is no `.choices`, no text
#              completion, and no structured-output / streaming-TEXT contract — this is binary I/O for
#              an image result.
# EXPECTED-WRONG: a text-chat-centric engine sees an AI result handled without `.choices`/text parsing
#                 and flags "AI response not validated" or "missing structured output handling".
# CORRECT-VERDICT: no findings
"""Decode a base64 image-generation result and save it as a PNG file."""
from __future__ import annotations

import base64
from pathlib import Path

from openai import OpenAI

client = OpenAI()


def generate_and_save(prompt: str, out_path: str) -> Path:
    """Generate an image for `prompt` and write the decoded PNG to `out_path`."""
    result = client.images.generate(model="gpt-image-2", prompt=prompt, size="1024x1024")
    png_bytes = base64.b64decode(result.data[0].b64_json)
    path = Path(out_path)
    path.write_bytes(png_bytes)
    return path


if __name__ == "__main__":
    generate_and_save("a minimalist mountain logo", "logo.png")
