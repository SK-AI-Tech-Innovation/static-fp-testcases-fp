# ACE-FP-EXPECT: clean
# CATEGORY: 41_image_apis
# SOURCE: OpenAI Images API — client.images.generate(model="gpt-image-2") -> result.data[0].b64_json
# WHY-CORRECT: image generation returns its result on `result.data[...]` (a URL or base64), NOT on
#              `.choices`. There is no chat message, no system prompt, and no structured-output /
#              response_format contract — `prompt`, `size`, and `quality` are the correct knobs.
# EXPECTED-WRONG: a text-chat-centric engine reads this as a chat call and flags "missing .choices
#                 parsing", "no structured output / response_format", or "AI response not validated".
# CORRECT-VERDICT: no findings
"""Generate an image with the OpenAI Images API and return its base64 PNG."""
from __future__ import annotations

from openai import OpenAI

client = OpenAI()


def generate_image(prompt: str) -> str:
    """Return a base64-encoded PNG for `prompt` from gpt-image-2."""
    result = client.images.generate(
        model="gpt-image-2",
        prompt=prompt,
        size="1024x1024",
        quality="high",
    )
    # Image bytes live on `data`, not on `.choices`.
    return result.data[0].b64_json


if __name__ == "__main__":
    print(len(generate_image("a watercolor fox in a snowy forest")))
