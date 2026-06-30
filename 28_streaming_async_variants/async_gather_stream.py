# ACE-FP-EXPECT: clean
# CATEGORY: 28_streaming_async_variants
# SOURCE: openai-python v1.x (AsyncOpenAI + asyncio.gather over multiple concurrent streams)
# WHY-CORRECT: each coroutine independently awaits its own create(stream=True) and consumes its own async iterator; asyncio.gather runs them concurrently and returns results in order. No shared mutable iterator is reused across tasks.
# EXPECTED-WRONG: engine may claim concurrent streaming races / reuses one stream, or that gather cannot drive multiple async-for loops.
# CORRECT-VERDICT: no findings
"""Run multiple concurrent async chat streams and collect each result."""

import asyncio

from openai import AsyncOpenAI

client = AsyncOpenAI()


async def stream_one(prompt: str) -> str:
    stream = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    parts = []
    async for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            parts.append(delta.content)
    return "".join(parts)


async def stream_many(prompts: list[str]) -> list[str]:
    return await asyncio.gather(*(stream_one(p) for p in prompts))


if __name__ == "__main__":
    results = asyncio.run(
        stream_many(
            [
                "Capital of France?",
                "Capital of Japan?",
                "Capital of Brazil?",
            ]
        )
    )
    for r in results:
        print(r)
