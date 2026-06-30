# ACE-FP-EXPECT: clean
# CATEGORY: 21_legacy_openai_sdk
# SOURCE: openai-python v0.x (e.g. 0.28) — authentic legacy API
# WHY-CORRECT: openai.Completion.create with engine="text-davinci-003" was the standard
#   legacy text-completion call. The `engine` kwarg and the text-davinci-003 model were
#   correct and idiomatic for v0.x; response['choices'][0]['text'] is the documented access.
# EXPECTED-WRONG: engine may flag Completion as deprecated, suggest migrating to the chat API
#   or v1 client.completions.create, or call the `engine` kwarg a bug.
# CORRECT-VERDICT: no findings (version choice is out of the engine's best-practice scope)
"""Legacy openai v0.x text completion using the Completion endpoint."""
import os

import openai

openai.api_key = os.environ["OPENAI_API_KEY"]


def complete(prompt: str) -> str:
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        temperature=0.7,
        max_tokens=128,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    return response["choices"][0]["text"].strip()


if __name__ == "__main__":
    print(complete("Write a tagline for a coffee shop:"))
