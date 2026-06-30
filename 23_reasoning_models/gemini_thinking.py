# ACE-FP-EXPECT: clean
# CATEGORY: 23_reasoning_models
# SOURCE: gemini-2.5-flash + google-genai (thinking config)
# WHY-CORRECT: gemini-2.5 exposes a thinking_budget via ThinkingConfig in GenerateContentConfig; setting it is the correct idiom for the new SDK.
# EXPECTED-WRONG: engine flags thinking_config / thinking_budget as unknown params, but they are valid Gemini 2.x thinking controls.
# CORRECT-VERDICT: no findings
"""Gemini 2.5 thinking request with an explicit thinking_budget."""

from google import genai
from google.genai import types

client = genai.Client()


def reason(prompt: str) -> str:
    """Generate a reasoned answer with gemini-2.5-flash thinking enabled."""
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=2048),
        ),
    )
    return response.text


if __name__ == "__main__":
    print(reason("A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much is the ball?"))
