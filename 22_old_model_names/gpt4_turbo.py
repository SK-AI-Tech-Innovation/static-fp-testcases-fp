# ACE-FP-EXPECT: clean
# CATEGORY: 22_old_model_names
# SOURCE: gpt-4-1106-preview (gpt-4-turbo) + openai-python
# WHY-CORRECT: gpt-4-1106-preview is a valid gpt-4-turbo snapshot; the chat.completions call with JSON response_format is correctly formed.
# EXPECTED-WRONG: engine flags "preview/turbo model is dated, use gpt-4o" as a finding, but the call is valid and model choice is not a best-practice defect.
# CORRECT-VERDICT: no findings
"""Structured extraction with gpt-4-turbo (gpt-4-1106-preview)."""

import json

from openai import OpenAI

client = OpenAI()


def extract_contact(text: str) -> dict:
    """Extract name and email as JSON from a freeform message."""
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "Extract the contact as JSON with keys 'name' and 'email'."},
            {"role": "user", "content": text},
        ],
        temperature=0,
    )
    return json.loads(response.choices[0].message.content)


if __name__ == "__main__":
    print(extract_contact("Hi, I'm Dana Lee, reach me at dana@example.com."))
