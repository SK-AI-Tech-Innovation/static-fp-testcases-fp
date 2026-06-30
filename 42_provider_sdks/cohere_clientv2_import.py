# ACE-FP-EXPECT: clean
# CATEGORY: 42_provider_sdks
# SOURCE: Cohere Python SDK, verified June 2026
# WHY-CORRECT: cohere.ClientV2 is the current entrypoint; chat takes a messages=[...] list and returns res.message.content[0].text
# EXPECTED-WRONG: stale analyzer only knows the legacy cohere.Client().chat(message=...) shape and flags ClientV2 + messages=[...] as an unknown/malformed SDK usage
# CORRECT-VERDICT: no findings
"""Construct cohere.ClientV2 and chat with the messages-list interface."""

import os

import cohere

co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))


def main() -> None:
    res = co.chat(
        model="command-a-plus-05-2026",
        messages=[{"role": "user", "content": "Name a constellation in the night sky."}],
    )
    print(res.message.content[0].text)


if __name__ == "__main__":
    main()
