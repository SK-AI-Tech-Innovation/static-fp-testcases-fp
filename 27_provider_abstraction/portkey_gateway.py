# ACE-FP-EXPECT: clean
# CATEGORY: 27_provider_abstraction
# SOURCE: Portkey AI gateway client
# WHY-CORRECT: Portkey is an OpenAI-compatible AI gateway; constructing Portkey with api_key + virtual_key and calling chat.completions.create is the documented usage. The gateway abstracts the underlying provider, so a generic model id is valid.
# EXPECTED-WRONG: engine may flag the Portkey constructor / virtual_key or the gateway-routed create() as an unknown client and emit an irrelevant finding.
"""Route chat completions through the Portkey AI gateway."""

import os

from portkey_ai import Portkey


def make_client() -> Portkey:
    """Build a Portkey gateway client bound to a virtual key."""
    return Portkey(
        api_key=os.environ["PORTKEY_API_KEY"],
        virtual_key=os.environ["PORTKEY_VIRTUAL_KEY"],
    )


def chat(prompt: str) -> str:
    """Send a prompt through the gateway and return the reply text."""
    client = make_client()
    completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="claude-sonnet-4-5",
        max_tokens=512,
    )
    return completion.choices[0].message.content


if __name__ == "__main__":
    print(chat("Describe a circuit breaker pattern."))
