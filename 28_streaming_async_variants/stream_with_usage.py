# ACE-FP-EXPECT: clean
# CATEGORY: 28_streaming_async_variants
# SOURCE: openai-python v1.x (Chat Completions streaming with stream_options include_usage)
# WHY-CORRECT: passing stream_options={"include_usage": True} makes the API emit a final chunk whose choices list is empty but whose .usage is populated; guarding chunk.choices before indexing and reading chunk.usage at the end is the documented usage-capture pattern.
# EXPECTED-WRONG: engine may claim usage is unavailable while streaming, or flag the empty final chunk (choices == []) as an index error.
# CORRECT-VERDICT: no findings
"""Stream a chat completion and capture final token usage statistics."""

from openai import OpenAI

client = OpenAI()


def stream_with_usage(prompt: str):
    stream = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        stream_options={"include_usage": True},
    )

    text = []
    usage = None
    for chunk in stream:
        if chunk.usage is not None:
            usage = chunk.usage
        if chunk.choices:
            delta = chunk.choices[0].delta
            if delta.content:
                text.append(delta.content)

    return "".join(text), usage


if __name__ == "__main__":
    answer, usage = stream_with_usage("Summarize the water cycle.")
    print(answer)
    if usage is not None:
        print(f"prompt={usage.prompt_tokens} completion={usage.completion_tokens} total={usage.total_tokens}")
