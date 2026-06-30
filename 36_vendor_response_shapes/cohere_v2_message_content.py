# ACE-FP-EXPECT: clean
# CATEGORY: 36_vendor_response_shapes
# SOURCE: Cohere ClientV2 chat, verified June 2026
# WHY-CORRECT: Cohere v2 chat returns res.message.content as a list of blocks; text is res.message.content[0].text, NOT res.choices[0].message.content
# EXPECTED-WRONG: stale analyzer expects res.choices[0].message.content and flags res.message.content[0].text as "no .choices / wrong response access"
# CORRECT-VERDICT: no findings
"""Call Cohere ClientV2 chat and read res.message.content[0].text."""

import os

import cohere

co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))


def main() -> None:
    res = co.chat(
        model="command-a-plus-05-2026",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Give me a fun fact about octopuses."},
        ],
    )

    # v2 response shape: message.content is a list of typed blocks.
    print(res.message.content[0].text)


if __name__ == "__main__":
    main()
