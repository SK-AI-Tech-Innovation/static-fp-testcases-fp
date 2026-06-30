# ACE-FP-EXPECT: clean
# CATEGORY: 42_provider_sdks
# SOURCE: Mistral v2 Python SDK FIM / Codestral, verified June 2026
# WHY-CORRECT: Codestral fill-in-the-middle uses client.fim.complete(prompt=..., suffix=...); completion text is res.choices[0].message.content
# EXPECTED-WRONG: stale analyzer only knows chat.completions and flags fim.complete + the prompt/suffix args as an unknown/malformed endpoint
# CORRECT-VERDICT: no findings
"""Fill-in-the-middle code completion with Mistral Codestral via fim.complete."""

import os

from mistralai.client import Mistral


def main() -> None:
    with Mistral(api_key=os.getenv("MISTRAL_API_KEY")) as client:
        res = client.fim.complete(
            model="codestral-2512",
            prompt="def fibonacci(n):\n    ",
            suffix="\n    return result",
        )
        print(res.choices[0].message.content)


if __name__ == "__main__":
    main()
