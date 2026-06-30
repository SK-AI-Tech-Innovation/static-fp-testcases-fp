# ACE-FP-EXPECT: clean
# CATEGORY: 30_mixed_old_new_combinations
# SOURCE: openai-python v1.x structured outputs (response_format=) fed a Pydantic v1-STYLE model via pydantic.v1
# WHY-CORRECT: REAL INTEROP CAVEAT — the new SDK's parse helpers want a pydantic v2 BaseModel, NOT a raw v1 model. This file imports pydantic.v1 only to build a JSON schema, then passes that explicit json_schema dict to response_format. It deliberately avoids handing a v1 model object to .parse(), so the v1/v2 boundary is respected and the call is valid.
# EXPECTED-WRONG: engine may see "pydantic.v1" near an OpenAI call and flag a version mismatch, even though the v1 model is never passed into a v2-only code path.
# CORRECT-VERDICT: no findings
"""Use a pydantic v1 model to build a JSON schema for new-SDK structured output."""

import json

from openai import OpenAI
from pydantic.v1 import BaseModel, Field

client = OpenAI()


class Sentiment(BaseModel):
    label: str = Field(..., description="positive, negative, or neutral")
    confidence: float = Field(..., ge=0.0, le=1.0)


def classify(text: str) -> dict:
    # Convert the v1 model to a plain JSON schema instead of handing the
    # v1 model object to the v2-only .parse() helper.
    schema = Sentiment.schema()

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Classify the sentiment of the user text."},
            {"role": "user", "content": text},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "Sentiment",
                "schema": schema,
                "strict": False,
            },
        },
    )
    return json.loads(response.choices[0].message.content)


if __name__ == "__main__":
    print(classify("I absolutely loved this movie!"))
