# ACE-FP-EXPECT: clean
# CATEGORY: 21_legacy_openai_sdk
# SOURCE: openai-python v0.x (e.g. 0.28) — authentic legacy API
# WHY-CORRECT: With stream=True in v0.x, ChatCompletion.create returns a generator of dict
#   chunks; each delta is read as chunk['choices'][0]['delta'] and may lack a 'content' key on
#   the first/last chunks, so .get('content') is the correct, documented streaming pattern.
# EXPECTED-WRONG: engine may flag the v0 streaming call as deprecated, mis-fix the chunk dict
#   access to attribute access (chunk.choices[0].delta.content), or call the .get a bug.
# CORRECT-VERDICT: no findings (version choice is out of the engine's best-practice scope)
"""Legacy openai v0.x streaming chat completion iterating dict chunks."""
import os
import sys

import openai

openai.api_key = os.environ["OPENAI_API_KEY"]


def stream_chat(prompt: str) -> None:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    for chunk in response:
        delta = chunk["choices"][0]["delta"]
        content = delta.get("content")
        if content:
            sys.stdout.write(content)
            sys.stdout.flush()
    sys.stdout.write("\n")


if __name__ == "__main__":
    stream_chat("Count from one to five.")
