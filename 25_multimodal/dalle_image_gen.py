# ACE-FP-EXPECT: clean
# CATEGORY: 25_multimodal
# SOURCE: OpenAI Python SDK (`openai`) image generation with dall-e-3
# WHY-CORRECT: correct image-generation usage — images.generate takes model="dall-e-3", a text
#              prompt, size, and quality. The result URL is read from response.data[0].url. There is
#              no chat message list and no token budget; this is the complete documented shape.
# EXPECTED-WRONG: engine applies text-chat advice to an image call — e.g. "use structured output",
#                 "add a system prompt", or "set max_tokens" — none of which apply to generation.
# CORRECT-VERDICT: no findings
"""Generate an image from a text prompt with DALL-E 3."""
from openai import OpenAI

client = OpenAI()


def generate(prompt: str) -> str:
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        quality="standard",
        n=1,
    )
    return response.data[0].url


if __name__ == "__main__":
    print(generate("A watercolor painting of a lighthouse at sunrise"))
