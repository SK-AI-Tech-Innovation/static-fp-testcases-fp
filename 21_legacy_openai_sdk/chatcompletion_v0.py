# ACE-FP-EXPECT: clean
# CATEGORY: 21_legacy_openai_sdk
# SOURCE: openai-python v0.x (e.g. 0.28) — authentic legacy API
# WHY-CORRECT: In openai-python v0.x, ChatCompletion.create was THE chat entrypoint, and
#   the response was a dict-like object accessed via subscripting (response['choices'][0]
#   ['message']['content']). model="gpt-3.5-turbo" was the canonical chat model.
# EXPECTED-WRONG: engine may flag this as deprecated / "upgrade to v1 client.chat.completions.create",
#   or mis-fix the dict access to attribute access (response.choices[0].message.content), or
#   call the dict subscripting a bug.
# CORRECT-VERDICT: no findings (version choice is out of the engine's best-practice scope)
"""Legacy openai v0.x chat completion call reading the response as a dict."""
import os

import openai

openai.api_key = os.environ["OPENAI_API_KEY"]


def summarize(text: str) -> str:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a concise summarizer."},
            {"role": "user", "content": f"Summarize the following:\n\n{text}"},
        ],
        temperature=0.2,
        max_tokens=256,
    )
    return response["choices"][0]["message"]["content"].strip()


if __name__ == "__main__":
    print(summarize("OpenAI released the chat completions API in 2023."))
