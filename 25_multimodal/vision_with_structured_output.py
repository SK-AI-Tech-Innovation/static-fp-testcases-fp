# ACE-FP-EXPECT: clean
# CATEGORY: 25_multimodal
# SOURCE: OpenAI Python SDK (`openai`) Responses API vision input + Pydantic structured output
# WHY-CORRECT: combines a multimodal image input with structured output — input mixes input_text +
#              input_image parts, and responses.parse enforces a Pydantic schema via text_format.
#              The parsed object is read from response.output_parsed. Both idioms already applied.
# EXPECTED-WRONG: engine still fires "use structured output" even though text_format=ReceiptFields
#                 is present, or rejects the input_image part and suggests a plain text prompt.
# CORRECT-VERDICT: no findings
"""Extract structured fields from a receipt image with the Responses API."""
from openai import OpenAI
from pydantic import BaseModel

client = OpenAI()


class ReceiptFields(BaseModel):
    merchant: str
    total: float
    currency: str
    date: str


def extract(image_url: str) -> ReceiptFields:
    response = client.responses.parse(
        model="gpt-4o",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Extract the receipt fields from this image."},
                    {"type": "input_image", "image_url": image_url},
                ],
            }
        ],
        text_format=ReceiptFields,
    )
    return response.output_parsed


if __name__ == "__main__":
    print(extract("https://example.com/receipt.png"))
