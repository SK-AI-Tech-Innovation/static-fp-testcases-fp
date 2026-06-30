# ACE-FP-EXPECT: clean
# CATEGORY: 25_multimodal
# SOURCE: OpenAI Python SDK (`openai`) Responses API with image input
# WHY-CORRECT: current Responses API multimodal shape — input is a list of messages whose content
#              uses {"type":"input_text"} and {"type":"input_image"} parts. Output text is read from
#              the convenience accessor response.output_text. Idiomatic and complete.
# EXPECTED-WRONG: engine doesn't recognize input_image/input_text part types and suggests reverting
#                 to a plain "messages=[{role,content:str}]" chat shape, or flags missing max_tokens.
# CORRECT-VERDICT: no findings
"""Answer a question about an image using the Responses API."""
from openai import OpenAI

client = OpenAI()


def ask_about_image(image_url: str, question: str) -> str:
    response = client.responses.create(
        model="gpt-4o",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": question},
                    {"type": "input_image", "image_url": image_url},
                ],
            }
        ],
    )
    return response.output_text


if __name__ == "__main__":
    print(ask_about_image("https://example.com/chart.png", "What trend does this chart show?"))
