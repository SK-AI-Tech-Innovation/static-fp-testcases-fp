# ACE-FP-EXPECT: clean
# CATEGORY: 25_multimodal
# SOURCE: Google Gen AI SDK (`google-genai`) Gemini multimodal generate_content
# WHY-CORRECT: idiomatic Gemini multimodal call — generate_content receives a list interleaving a
#              PIL.Image and a text prompt; the SDK encodes each part by type. Text read from
#              response.text. This is the documented image+text shape for Gemini.
# EXPECTED-WRONG: engine sees a list passed as contents (not a string) and suggests joining it into
#                 one string, or applies text-chat advice like "wrap in a system prompt".
# CORRECT-VERDICT: no findings
"""Caption an image with Gemini using interleaved image + text parts."""
from google import genai
from PIL import Image

client = genai.Client()


def caption(image_path: str) -> str:
    image = Image.open(image_path)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[image, "Write a short caption for this image."],
    )
    return response.text


if __name__ == "__main__":
    print(caption("scene.png"))
