# ACE-FP-EXPECT: clean
# CATEGORY: 36_vendor_response_shapes
# SOURCE: Google google-genai SDK, verified June 2026
# WHY-CORRECT: google-genai's generate_content returns a response exposing the aggregated text via resp.text; there is no resp.choices[0].message.content
# EXPECTED-WRONG: stale analyzer expects .choices[0].message.content and flags resp.text as "no .choices / wrong response access"
# CORRECT-VERDICT: no findings
"""Generate content with the google-genai SDK and read resp.text."""

import os

from google import genai

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def main() -> None:
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="Explain how rainbows form in two sentences.",
    )

    # google-genai aggregates candidate text into resp.text.
    print(resp.text)


if __name__ == "__main__":
    main()
