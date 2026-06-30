# ACE-FP-EXPECT: clean
# CATEGORY: 42_provider_sdks
# SOURCE: Mistral v2 Python SDK, verified June 2026
# WHY-CORRECT: the v2 SDK is used as a context manager (with Mistral(...) as client) and chat is client.chat.complete(...); response text is res.choices[0].message.content
# EXPECTED-WRONG: stale analyzer expects the legacy MistralClient().chat(...) call shape and flags the context manager + client.chat.complete(...) as a malformed/unknown SDK call
# CORRECT-VERDICT: no findings
"""Use the Mistral v2 SDK as a context manager and call chat.complete."""

import os

from mistralai.client import Mistral


def main() -> None:
    with Mistral(api_key=os.getenv("MISTRAL_API_KEY")) as client:
        res = client.chat.complete(
            model="mistral-large-2512",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the speed of light?"},
            ],
        )
        print(res.choices[0].message.content)


if __name__ == "__main__":
    main()
