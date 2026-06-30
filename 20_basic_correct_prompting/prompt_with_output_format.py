# ACE-FP-EXPECT: clean
# CATEGORY: 20_basic_correct_prompting
# SOURCE: OpenAI JSON-mode prompt validated with a Pydantic schema
# WHY-CORRECT: the prompt asks for JSON, response_format pins JSON output, and the result is parsed
#              AND validated against a Pydantic model. Belt-and-suspenders — both instruction and
#              schema enforcement are present, so nothing is missing.
# EXPECTED-WRONG: engine suggests "validate the output" or "request structured output" — both are
#                 already done here.
# CORRECT-VERDICT: no findings
"""Ask for JSON, enforce JSON mode, and validate with Pydantic."""
import json

from openai import OpenAI
from pydantic import BaseModel

client = OpenAI()


class Person(BaseModel):
    name: str
    age: int


def extract(text: str) -> Person:
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": 'Return JSON with keys "name" (string) and "age" (integer).',
            },
            {"role": "user", "content": text},
        ],
    )
    payload = json.loads(response.choices[0].message.content)
    return Person.model_validate(payload)


if __name__ == "__main__":
    print(extract("Alice is 30 years old."))
