# ACE-FP-EXPECT: clean
# CATEGORY: 25_multimodal
# SOURCE: OpenAI Python SDK (`openai`) GPT-4o vision via Chat Completions
# WHY-CORRECT: correct multimodal Chat Completions usage — a single user message whose content is
#              a list of typed parts mixing {"type":"text"} and {"type":"image_url"}. This is the
#              documented way to send an image to gpt-4o; reply read from choices[0].message.content.
# EXPECTED-WRONG: engine treats the image_url content part as malformed text content and suggests
#                 "pass content as a plain string" or flags "use structured output" on a vision call.
# CORRECT-VERDICT: no findings
"""Describe the contents of an image URL with GPT-4o vision."""
from openai import OpenAI

client = OpenAI()


def describe_image(image_url: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {"type": "image_url", "image_url": {"url": image_url, "detail": "auto"}},
                ],
            }
        ],
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    print(describe_image("https://example.com/cat.jpg"))
