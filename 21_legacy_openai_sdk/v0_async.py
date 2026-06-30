# ACE-FP-EXPECT: clean
# CATEGORY: 21_legacy_openai_sdk
# SOURCE: openai-python v0.x (e.g. 0.28) — authentic legacy API
# WHY-CORRECT: v0.x exposed async via the `acreate` coroutine method
#   (openai.ChatCompletion.acreate(...)). Awaiting it and reading the dict response is the
#   documented legacy async pattern; gathering several coroutines with asyncio is idiomatic.
# EXPECTED-WRONG: engine may flag acreate as deprecated, push the v1 AsyncOpenAI() client
#   (await client.chat.completions.create), or call acreate a bug / typo of create.
# CORRECT-VERDICT: no findings (version choice is out of the engine's best-practice scope)
"""Legacy openai v0.x async chat completion via ChatCompletion.acreate."""
import asyncio
import os

import openai

openai.api_key = os.environ["OPENAI_API_KEY"]


async def chat(prompt: str) -> str:
    response = await openai.ChatCompletion.acreate(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    return response["choices"][0]["message"]["content"]


async def main() -> None:
    prompts = ["Say one.", "Say two.", "Say three."]
    results = await asyncio.gather(*(chat(p) for p in prompts))
    for r in results:
        print(r)


if __name__ == "__main__":
    asyncio.run(main())
