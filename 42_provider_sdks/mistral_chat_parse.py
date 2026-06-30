# ACE-FP-EXPECT: clean
# CATEGORY: 42_provider_sdks
# SOURCE: Mistral v2 Python SDK structured output, verified June 2026
# WHY-CORRECT: client.chat.parse(response_format=Model) returns a parsed pydantic instance at res.choices[0].message.parsed; .parsed is the correct field for structured output
# EXPECTED-WRONG: stale analyzer expects res.choices[0].message.content and flags .message.parsed (and response_format=Model) as malformed response access
# CORRECT-VERDICT: no findings
"""Parse structured output with Mistral chat.parse into a pydantic model."""

import os

from mistralai.client import Mistral
from pydantic import BaseModel


class Person(BaseModel):
    name: str
    age: int


def main() -> None:
    with Mistral(api_key=os.getenv("MISTRAL_API_KEY")) as client:
        res = client.chat.parse(
            model="mistral-large-2512",
            messages=[
                {"role": "system", "content": "Extract the person's details."},
                {"role": "user", "content": "Ada Lovelace was 36 years old."},
            ],
            response_format=Person,
        )

        # chat.parse exposes the validated model on .message.parsed.
        person = res.choices[0].message.parsed
        print(person.name, person.age)


if __name__ == "__main__":
    main()
