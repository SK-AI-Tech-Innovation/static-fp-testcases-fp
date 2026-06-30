# ACE-FP-EXPECT: clean
# CATEGORY: 25_multimodal
# SOURCE: OpenAI Python SDK (`openai`) image generation with gpt-image-1
# WHY-CORRECT: current image model — images.generate with model="gpt-image-1" returns base64 image
#              data (b64_json), which is decoded and written to disk. Idiomatic for gpt-image-1,
#              which returns base64 rather than a URL by default. Complete and correct.
# EXPECTED-WRONG: engine flags gpt-image-1 as an unknown model id, or applies text-chat findings
#                 ("use structured output", "missing system prompt") to an image-generation call.
# CORRECT-VERDICT: no findings
"""Generate an image with gpt-image-1 and save the decoded PNG."""
import base64

from openai import OpenAI

client = OpenAI()


def generate(prompt: str, out_path: str = "out.png") -> str:
    response = client.images.generate(
        model="gpt-image-1",
        prompt=prompt,
        size="1024x1024",
    )
    image_bytes = base64.b64decode(response.data[0].b64_json)
    with open(out_path, "wb") as f:
        f.write(image_bytes)
    return out_path


if __name__ == "__main__":
    print(generate("An isometric illustration of a cozy bookstore"))
