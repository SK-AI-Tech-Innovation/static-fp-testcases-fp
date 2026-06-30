# ACE-FP-EXPECT: clean
# CATEGORY: 42_provider_sdks
# SOURCE: Google google-genai SDK, verified June 2026
# WHY-CORRECT: the current google-genai SDK uses genai.Client(api_key=...) and client.models.generate_content(model=..., contents=...); text is read from resp.text
# EXPECTED-WRONG: stale analyzer expects the deprecated google.generativeai GenerativeModel(...).generate_content or an OpenAI client and flags genai.Client().models.generate_content as malformed
# CORRECT-VERDICT: no findings
"""Use the google-genai Client.models.generate_content interface."""

import os

from google import genai
from google.genai import types

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def main() -> None:
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="List two benefits of unit testing.",
        config=types.GenerateContentConfig(
            system_instruction="You are a concise software mentor.",
            max_output_tokens=256,
        ),
    )
    print(resp.text)


if __name__ == "__main__":
    main()
