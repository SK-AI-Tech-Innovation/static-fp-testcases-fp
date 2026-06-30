# ACE-FP-EXPECT: clean
# CATEGORY: 41_image_apis
# SOURCE: Image generation via Replicate (replicate.run) and fal.ai (fal_client.subscribe) SDKs
# WHY-CORRECT: Replicate and fal run image models and return outputs as URLs/file descriptors from a
#              single SDK call (the SDKs handle the async lifecycle internally). These are image
#              endpoints: no `.choices`, no chat messages, no structured-output / streaming-TEXT contract,
#              and no hand-rolled unbounded polling — the client manages completion.
# EXPECTED-WRONG: a text-chat-centric engine treats these as chat calls and flags "no .choices",
#                 "AI response missing structured output", or "no retry/fallback on AI request".
# CORRECT-VERDICT: no findings
"""Generate images through the Replicate and fal.ai SDKs."""
from __future__ import annotations

import fal_client
import replicate


def generate_replicate(prompt: str) -> str:
    """Run a FLUX model on Replicate and return the first output URL."""
    output = replicate.run(
        "black-forest-labs/flux-1.1-pro",
        input={"prompt": prompt, "aspect_ratio": "1:1"},
    )
    # `output` is a list of image URLs/files for image models — never `.choices`.
    return str(output[0])


def generate_fal(prompt: str) -> str:
    """Run a FLUX model on fal.ai and return the resulting image URL."""
    result = fal_client.subscribe(
        "fal-ai/flux/dev",
        arguments={"prompt": prompt, "image_size": "square_hd"},
    )
    return result["images"][0]["url"]


if __name__ == "__main__":
    print(generate_replicate("a cozy reading nook by a rainy window"))
