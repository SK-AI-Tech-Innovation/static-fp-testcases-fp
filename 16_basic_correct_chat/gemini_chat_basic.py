# ACE-FP-EXPECT: clean
# CATEGORY: 16_basic_correct_chat
# SOURCE: Google Gen AI SDK (`google-genai`) `client.models.generate_content`
# WHY-CORRECT: current google-genai shape — Client() with model + contents, reply read from
#              response.text. This is the supported SDK (not the deprecated google-generativeai).
# EXPECTED-WRONG: engine flags "uses old google.generativeai" (it does not) or "add safety settings"
# CORRECT-VERDICT: no findings
"""Generate text with the Google Gen AI SDK."""
from google import genai

client = genai.Client()


def ask(prompt: str) -> str:
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    return response.text


if __name__ == "__main__":
    print(ask("Give me one fun fact about octopuses."))
