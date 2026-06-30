# ACE-FP-EXPECT: clean
# CATEGORY: 42_provider_sdks
# SOURCE: xAI native Python SDK (xai_sdk), verified June 2026
# WHY-CORRECT: the native xai_sdk uses Client() with client.chat.create(model=...), then appends messages and calls .sample(); it is not the OpenAI SDK shape
# EXPECTED-WRONG: stale analyzer expects from openai import OpenAI / chat.completions.create and flags Client().chat.create(...).sample() as an unknown/malformed SDK
# CORRECT-VERDICT: no findings
"""Chat with Grok through the native xai_sdk Client interface."""

import os

from xai_sdk import Client
from xai_sdk.chat import system, user


def main() -> None:
    client = Client(api_key=os.getenv("XAI_API_KEY"))

    chat = client.chat.create(model="grok-4.3")
    chat.append(system("You are Grok, a witty assistant."))
    chat.append(user("Tell me a one-line joke about astronomers."))

    response = chat.sample()
    print(response.content)


if __name__ == "__main__":
    main()
