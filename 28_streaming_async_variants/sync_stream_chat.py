# ACE-FP-EXPECT: clean
# CATEGORY: 28_streaming_async_variants
# SOURCE: openai-python v1.x (Chat Completions, sync streaming)
# WHY-CORRECT: stream=True returns an iterator of ChatCompletionChunk objects; iterating and reading choices[0].delta.content (guarding None) is the documented pattern.
# EXPECTED-WRONG: engine may claim the response must be awaited or parsed as JSON, or that delta.content is always present.
# CORRECT-VERDICT: no findings
"""Synchronous streaming chat completion that prints deltas as they arrive."""

from openai import OpenAI

client = OpenAI()


def stream_answer(question: str) -> str:
    stream = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": question}],
        stream=True,
    )

    collected = []
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content is not None:
            collected.append(delta.content)
            print(delta.content, end="", flush=True)

    print()
    return "".join(collected)


if __name__ == "__main__":
    stream_answer("Give me a one sentence summary of photosynthesis.")
