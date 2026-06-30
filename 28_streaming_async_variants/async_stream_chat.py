# ACE-FP-EXPECT: clean
# CATEGORY: 28_streaming_async_variants
# SOURCE: openai-python v1.x (AsyncOpenAI, async streaming)
# WHY-CORRECT: AsyncOpenAI.chat.completions.create(stream=True) is a coroutine returning an async iterator; awaiting it then `async for` over chunks is the documented async pattern.
# EXPECTED-WRONG: engine may flag "async for over a non-async object" or claim await is misplaced / cannot await and iterate.
# CORRECT-VERDICT: no findings
"""Asynchronous streaming chat completion using AsyncOpenAI."""

import asyncio

from openai import AsyncOpenAI

client = AsyncOpenAI()


async def stream_answer(question: str) -> str:
    stream = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": question}],
        stream=True,
    )

    collected = []
    async for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            collected.append(delta.content)

    return "".join(collected)


if __name__ == "__main__":
    print(asyncio.run(stream_answer("Name three primary colors.")))
